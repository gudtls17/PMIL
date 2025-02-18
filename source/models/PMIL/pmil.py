from os.path import join
import json
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from typing import List
from .components import InterpretableTransformerEncoder, InterpretableTransformerDecoder
from omegaconf import DictConfig
from ..base import BaseModel
import pickle
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

class TransTransitionEncoder(nn.Module):
    """
    Transformer encoder with Brain Connectome graph transition mechanism.
    Input size: (batch, input_node_num, input_feature_size)
    OUtput size: (batch, input_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size,nHead=4):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=nHead, dim_feedforward=hidden_size, batch_first=True)

        self.class_token = nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True).cuda()
        self.reset_parameters()
        
    def reset_parameters(self):
        self.class_token = nn.init.xavier_normal_(self.class_token)

    def forward(self, x: torch.tensor, init_cls=True):
        bz, node_num, dim = x.shape
        
        if init_cls:
            """Add new cls token"""
            class_token = self.class_token
            class_token = class_token.repeat(bz,1,1)
            x = torch.cat((class_token, x), dim=1)  # (batch, input_node_num+1, input_feature_size)
        x = self.transformer(x)
    
        cls_token = x[:, 0, :]
        x = x[:, 1:, :]
        return x, None, cls_token.reshape(x.shape[0], 1, -1)

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

class TransTransitionDecoder(nn.Module):
    """
    Transformer decoder with Brain Connectome graph transition mechanism.
    Input size: (batch, CLS + input_node_num, input_feature_size)
    OUtput size: (batch, CLS + projected_input_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, nHead=4):
        super().__init__()
        self.transformer = InterpretableTransformerDecoder(d_model=input_feature_size, nhead=nHead, dim_feedforward=hidden_size, batch_first=True)
        
        self.class_token = nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True).cuda()
        self.reset_parameters()
        
    def reset_parameters(self):
        self.class_token = nn.init.xavier_normal_(self.class_token)

    def forward(self, x_query: torch.tensor, x_keyval: torch.tensor, init_cls=True):
        bz, node_num, dim = x_query.shape
        
        if init_cls:
            """Add new cls token"""
            class_token = self.class_token
            class_token = class_token.repeat(bz,1,1)
            x_query = torch.cat((class_token, x_query), dim=1)  # (batch, input_node_num+1, input_feature_size)
        
        x = self.transformer(x_query, x_keyval)
    
        cls_token = x[:, 0, :]
        x = x[:, 1:, :]
        return x, None, cls_token.reshape(x.shape[0], 1, -1)

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True, nHead=4):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=nHead, dim_feedforward=hidden_size,batch_first=True)
        
        self.pooling = pooling
        if self.pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)
        
        
    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x: torch.tensor):
        bz, node_num, dim = x.shape
        
        x = self.transformer(x)
        cls_token = x[:, 0, :]
        x = x[:, 1:, :]
        
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment, cls_token.reshape(x.shape[0], 1, -1)
        return x, None, cls_token.reshape(x.shape[0], 1, -1)

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)

class TransPoolingDecoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True, nHead=4):
        super().__init__()
        self.transformer = InterpretableTransformerDecoder(d_model=input_feature_size, nhead=nHead, dim_feedforward=hidden_size,  batch_first=True)
        
        self.pooling = pooling
        if self.pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x_query: torch.tensor, x_keyval: torch.tensor):
        
        x = self.transformer(x_query, x_keyval)
        cls_token = x[:, 0, :]
        x = x[:, 1:, :]
        
        if self.pooling:
            x, assignment = self.dec(x)  # exclude cls_token as input
            return x, assignment, cls_token.reshape(x.shape[0], 1, -1)
        return x, None, cls_token.reshape(x.shape[0], 1, -1)

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class Pmil(BaseModel):
    def __init__(self, config: DictConfig):

        super().__init__()
        
        """Load pretrained text model"""
        main_path = 'V:/XXX/Project/VisionLang2'
        
        model_BiomedCLIP, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')  # BiomedCLIP-PubMedBERT
        with open(join(main_path, "BiomedCLIP_tokenizer.json"), 'rt', encoding='UTF8') as f:  # load BiomedCLIP vocabulary list
            tokenizer_vocab = json.load(f)
        idxtoword = {v: k for k, v in tokenizer_vocab['model']['vocab'].items()}
        
        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        self.num_MHSA = config.model.num_MHSA
        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling 

        self.embedding_FC = nn.Sequential(
            nn.Linear(config.dataset.node_sz, forward_dim),
            nn.LeakyReLU()
        )
        self.embedding_TD = nn.Sequential(
            nn.Linear(config.dataset.node_sz, forward_dim),
            nn.LeakyReLU()
        )
        self.reconstruct_FC = nn.Sequential(
            nn.Linear(forward_dim, config.dataset.node_sz),
            nn.LeakyReLU()
        )    
        self.reconstruct_TD = nn.Sequential(
            nn.Linear(forward_dim, config.dataset.node_sz),
            nn.LeakyReLU()
        )    
        
        # self.fc2td_transformer_encoder = TransTransitionEncoder(input_feature_size=forward_dim, input_node_num=in_sizes[1], hidden_size=1024, nHead=config.model.nhead)
        self.fc2td_transformer = TransTransitionDecoder(input_feature_size=forward_dim, input_node_num=in_sizes[1], hidden_size=1024, nHead=config.model.nhead)
        # self.td2fc_transformer_encoder = TransTransitionEncoder(input_feature_size=forward_dim, input_node_num=in_sizes[1], hidden_size=1024, nHead=config.model.nhead)
        self.td2fc_transformer = TransTransitionDecoder(input_feature_size=forward_dim, input_node_num=in_sizes[1], hidden_size=1024, nHead=config.model.nhead)
        
        self.dim_reduction_1 = nn.Sequential(nn.Linear(forward_dim*2, forward_dim), nn.LeakyReLU())
        self.attention_list.append(TransPoolingEncoder(input_feature_size=forward_dim, input_node_num=in_sizes[1], hidden_size=1024, output_node_num=sizes[1],
                                                               pooling=True, orthogonal=config.model.orthogonal, freeze_center=config.model.freeze_center,
                                                               project_assignment=config.model.project_assignment, nHead=config.model.nhead))
        self.dim_reduction_2 = nn.Sequential(nn.Linear(forward_dim, 8), nn.LeakyReLU())
        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.99), requires_grad=True)
        
        # self.its_transformer = model_BiomedCLIP.text.transformer  # exclude CLS token pooler layer (BERT embedding + BERT encoder)
        self.its_BiomedCLIP_embedding = model_BiomedCLIP.text.transformer.embeddings  # BERT embedding layer
        self.its_BiomedCLIP_encoder = model_BiomedCLIP.text.transformer.encoder  # BERT encoder layer
        self.embedding_its = nn.Linear(768, config.dataset.node_sz)  # hidden dim [768] to 200
        self.act_gelu = nn.GELU()
        self.prompt_token = nn.Parameter(torch.Tensor(1, 768), requires_grad = True).cuda()
        self.reset_parameters()
        
        self.assignMat = None

    def reset_parameters(self):
        self.prompt_token = nn.init.xavier_normal_(self.prompt_token)
        
    def forward(self,connectivity: torch.tensor, timedelay: torch.tensor, timescales: torch.tensor):
        """
        connectivity: input FC (batch, input_node_num, input_feature_size)
        timedelay: input TD (batch, input_node_num, input_feature_size)
        timescales: input ITS (batch, input_word_num)
        
        input_FC -> embedding_FC -> latent_FC -> reconstructed TD
                                              -> fake embedding_TD -> fake latent_TD -> fake reconstructed FC
        
        input_TD -> embedding_TD -> latent_TD -> reconstructed FC
                                              -> fake embedding_FC -> fake latent_FC -> fake reconstructed TD
        """
        
        bz, _, _, = connectivity.shape

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            connectivity = torch.cat([connectivity, pos_emb], dim=-1)

        assignments = []
        attn_weights = []

        """ITS tokenizer to embedding"""
        timescales = self.its_BiomedCLIP_embedding(timescales) # tokenize to embedding -> (batch, [CLS]+ word_token, hidden dim [768])
        cls_token_its = timescales[:, 0, :]
        timescales = timescales[:, 1:, :]
        prompt_token = self.prompt_token.repeat(bz,2,1)  # learnable prompt token -> (batch, 2, hidden dim [768])
        
        timescales = torch.cat((cls_token_its.unsqueeze(1), prompt_token, timescales), dim=1)  # add prompt token -> (batch, [CLS] + prompt + word_token, hidden dim [768])
        timescales = self.its_BiomedCLIP_encoder(timescales, output_attentions=True).last_hidden_state # embedding encoder -> (batch, [CLS]+ prompt+ word_token, hidden dim [768])
        timescales = self.act_gelu(self.embedding_its(timescales)) # (Batch, [CLS]+ prompt+ word_token, hidden dim [768]) -> (Batch, [CLS]+ prompt+ word_token, 200)
        
        """FC to TD"""
        # embed_FC, _, cls_token_FC = self.fc2td_transformer_encoder(connectivity, init_cls=True)  # (batch, input_node_num, input_feature_size) -> (batch, input_node_num+1, input_feature_size)
        embed_FC, _, cls_token_FC = self.fc2td_transformer(connectivity, timescales, init_cls=True) # query: embed_FC, key and value: timescales
        recon_timedelay = self.reconstruct_TD(torch.cat((cls_token_FC, embed_FC), dim=1))  # reconstructed timedelay matrix (batch, input_node_num+1, input_feature_size)
        
        # cycle_embed_TD, _, cycle_cls_token_FC = self.td2fc_transformer_encoder(torch.cat((cls_token_FC, embed_FC), dim=1), init_cls=False)  # (batch, input_node_num+1, input_feature_size) -> (batch, input_node_num+1, input_feature_size)
        cycle_embed_FC, _, cycle_cls_token_FC = self.td2fc_transformer(torch.cat((cls_token_FC, embed_FC), dim=1), timescales, init_cls=False) # query: cycle_embed_TD, key and value: timescales
        cycle_connectivity = self.reconstruct_FC(torch.cat((cycle_cls_token_FC, cycle_embed_FC), dim=1))  # cycle connectivity matrix (batch, input_node_num+1, input_feature_size)

        """TD to FC"""
        # embed_TD, _, cls_token_TD = self.td2fc_transformer_encoder(timedelay, init_cls=True)  # (batch, input_node_num, input_feature_size) -> (batch, input_node_num+1, input_feature_size)
        embed_TD, _, cls_token_TD = self.td2fc_transformer(timedelay, timescales, init_cls=True) # query: embed_FC, key and value: timescales
        recon_connectivity = self.reconstruct_FC(torch.cat((cls_token_TD, embed_TD), dim=1))  # reconstructed connectivity matrix (batch, input_node_num+1, input_feature_size)
        
        # cycle_embed_FC, _, cycle_cls_token_TD = self.fc2td_transformer_encoder(torch.cat((cls_token_TD, embed_TD), dim=1), init_cls=False)  # (batch, input_node_num+1, input_feature_size) -> (batch, input_node_num+1, input_feature_size)
        cycle_embed_TD, _, cycle_cls_token_TD = self.fc2td_transformer(torch.cat((cls_token_TD, embed_TD), dim=1), timescales, init_cls=False) # query: cycle_embed_TD, key and value: timescales
        cycle_timedelay = self.reconstruct_TD(torch.cat((cycle_cls_token_TD, cycle_embed_TD), dim=1))  # cycle timedelay matrix (batch, input_node_num+1, input_feature_size)

        """Merge FC and TD embedding"""
        embed_FC = torch.cat((cls_token_FC, embed_FC), dim=1)
        embed_TD = torch.cat((cls_token_TD, embed_TD), dim=1)
        alpha = torch.sigmoid(self.alpha)
        embed_merge = alpha*embed_FC + (1-alpha)*embed_TD  # mix with leanable portion
        # embed_merge = embed_FC + embed_TD  # SUM
        # embed_merge = (embed_FC + embed_TD)/2  # Mean
        # embed_merge = torch.max(torch.concat((embed_FC.unsqueeze(-1), embed_TD.unsqueeze(-1)), dim=-1), dim=-1)[0]  # MAX
        # embed_merge = torch.cat((embed_FC, embed_TD), dim=2)  # CONCAT (batch, input_node_num+1, input_feature_size*2)
        
        # embed_merge = self.dim_reduction_1(embed_merge)  # (batch, input_node_num+1, input_feature_size*2) -> (batch, input_node_num+1, input_feature_size)
        embed_merge, assign, cls_token_merge= self.attention_list[0](embed_merge) # pooling decoder, query: cls+embed_merge, key and value: timescales
        assignments.append(assign)
        attn_weights.append(self.attention_list[0].get_attention_weights())
        
        self.assignMat = assignments[0]
        
        embed_merge = self.dim_reduction_2(embed_merge)  # (batch, cluster_num, input_feature_size) -> (batch, cluster_num, 8)
        embed_merge = embed_merge.reshape((bz, -1))  # (batch, cluster_num, 8) -> (batch, cluster_num*8)
        
        return self.fc(embed_merge), recon_connectivity[:, 1:, :], recon_timedelay[:, 1:, :], cycle_connectivity[:, 1:, :], cycle_timedelay[:, 1:, :], cls_token_merge
        # return self.fc(embed_merge), recon_connectivity[:, 1:, :], None, None, None, None
        # return self.fc(embed_merge), None, recon_timedelay[:, 1:, :], None, None, None
        # return self.fc(embed_merge), recon_connectivity[:, 1:, :], recon_timedelay[:, 1:, :], None, None, None
        # return self.fc(embed_merge), None, None, None, None, None

    def get_assign_mat(self):
        return self.assignMat

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_local_attention_weights(self):
        return self.local_transformer.get_attention_weights()

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all
