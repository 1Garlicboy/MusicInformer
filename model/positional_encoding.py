import torch
import torch.nn as nn
import math
import typing as tp

# PositionalEncoding
# Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# 绝对位置编码
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 相对位置编码
# class RelativePositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1,max_len=5000) -> None:
#         super(RelativePositionalEncoding).__init__()
#         self.dropout=nn.Dropout(p=dropout)

#         pe=torch.zeros(max_len,d_model)
#         for pos in range (max_len):
#             for i in range(0,d_model,2):
#                 pe[pos,i]=torch.sin(pos/10000**(i/d_model))
#                 pe[pos,i+1]=torch.cos(pos/10000**(i/d_model))
            
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(RelativePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.max_len = max_len
        self.d_model = d_model

        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        

        # 计算相对位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        for pos in range (max_len):
            for i in range(0,d_model,2):
                pe[pos,i]=torch.sin(torch.tensor(pos/10000**(i/d_model)))
                pe[pos,i+1]=torch.cos(torch.tensor(pos/10000**(i/d_model)))
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 获取输入序列的长度
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
       
# class RotationalEncoding(nn.Module):

#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(RotationalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         # 生成欧拉角作为旋转位置编码
#         angles = torch.zeros(max_len, 3)  # 3个欧拉角，分别表示绕x、y、z轴的旋转
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         position = position.expand(max_len, 3)
#         angles[:, 0] = torch.sin(position * 0.01)  # 绕x轴的旋转
#         angles[:, 1] = torch.cos(position * 0.02)  # 绕y轴的旋转
#         angles[:, 2] = torch.sin(position * 0.03)  # 绕z轴的旋转
#         angles = angles.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('angles', angles)

#     def forward(self, x):
#         # 将输入序列与旋转位置编码相加
#         # x = x + self.angles[:x.size(0), :]
#         # return self.dropout(x)
#         seq_len = x.size(0)
#         angles = self.angles[:, :seq_len, :].to(x.device)
#         x = x + angles
#         return self.dropout(x)
    
# class RotationalEncoding(nn.Module):

#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(RotationalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         # 生成旋转矩阵作为旋转位置编码
#         angles = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) / max_len
#         angles = 2 * torch.pi * angles
#         self.register_buffer('angles', angles)

#     def forward(self, x):
#         # 将输入序列与旋转位置编码相加
#         seq_len = x.size(0)
#         angles = self.angles[:seq_len, :].unsqueeze(-1).unsqueeze(-1).to(x.device)
#         rotation_matrix_x = torch.cat([torch.ones_like(angles), torch.zeros_like(angles), torch.zeros_like(angles),
#                                        torch.zeros_like(angles), torch.cos(angles), -torch.sin(angles),
#                                        torch.zeros_like(angles), torch.sin(angles), torch.cos(angles)], dim=-1).view(-1, 3, 3)
#         rotation_matrix_y = torch.cat([torch.cos(angles), torch.zeros_like(angles), torch.sin(angles),
#                                        torch.zeros_like(angles), torch.ones_like(angles), torch.zeros_like(angles),
#                                        -torch.sin(angles), torch.zeros_like(angles), torch.cos(angles)], dim=-1).view(-1, 3, 3)
#         rotation_matrix_z = torch.cat([torch.cos(angles), -torch.sin(angles), torch.zeros_like(angles),
#                                        torch.sin(angles), torch.cos(angles), torch.zeros_like(angles),
#                                        torch.zeros_like(angles), torch.zeros_like(angles), torch.ones_like(angles)], dim=-1).view(-1, 3, 3)
#         rotation_matrix = torch.matmul(rotation_matrix_z, torch.matmul(rotation_matrix_y, rotation_matrix_x))
#         x = x.unsqueeze(-2)  # (seq_len, batch_size, d_model) -> (seq_len, 1, batch_size, d_model)
#         x=x.permute(0,1,3,2)
#         x = torch.matmul(x, rotation_matrix)  # 旋转输入序列
#         x = x.squeeze(-2)  # (seq_len, 1, batch_size, d_model) -> (seq_len, batch_size, d_model)
#         return self.dropout(x)

class RotationalEncoding(nn.Module):
    def __init__(self, d_model,dropout=0.1, max_len=5000):
        super(RotationalEncoding, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)

        # 计算旋转位置编码
        for pos in range(max_len):
            for i in range(d_model):
                angle = pos / max_len * (2 * i / d_model) * math.pi
                pe[pos, i] = math.sin(angle)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # 获取输入序列的长度
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)
    
# class RotaryEmbedding(torch.nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#         self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    
#     def forward(self, max_seq_len):
#         t = torch.arange(max_seq_len).type_as(self.inv_freq)
#         freqs = torch.einsum('i,j->ij', t, self.inv_freq)
#         emb = torch.cat((freqs, freqs), dim=-1)
#         return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Fundamental_Music_Embedding(nn.Module):
	def __init__(self, d_model, base, if_trainable = False, if_translation_bias_trainable = True, device='cuda', type = "se",emb_nn=None,translation_bias_type = "nd"):
		super().__init__()
		self.d_model = d_model
		self.device = device
		self.base = base
		self.if_trainable = if_trainable #whether the se is trainable 
		
		if translation_bias_type is not None:
			self.if_translation_bias = True
			self.if_translation_bias_trainable = if_translation_bias_trainable #default the 2d vector is trainable
			if translation_bias_type=="2d":
				translation_bias = torch.rand((1, 2), dtype = torch.float32) #Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)[0,1)
			elif translation_bias_type=="nd":
				translation_bias = torch.rand((1, self.d_model), dtype = torch.float32)
			translation_bias = nn.Parameter(translation_bias, requires_grad=True)
			self.register_parameter("translation_bias", translation_bias)
		else:
			self.if_translation_bias = False

		i = torch.arange(d_model)
		angle_rates = 1 / torch.pow(self.base, (2 * (i//2)) / d_model)
		angle_rates = angle_rates[None, ... ].to(self.device)

		if self.if_trainable:
			angles = nn.Parameter(angle_rates, requires_grad=True)
			self.register_parameter("angles", angles)
		
		else:
			self.angles = angle_rates

	def transform_by_delta_pos_v1(self, inp, delta_pos):
		#outdated version, use block diagonal matrix very inefficient
		if inp.dim()==3:
			batch, length = int(inp.shape[0]), int(inp.shape[1])
		elif inp.dim()==1:
			batch, length = 1, int(inp.shape[0])

		raw = self.FMS(delta_pos)
		
		wk_phi_1 = torch.reshape(raw,[batch, length,int(self.d_model/2), 2]) #[d_mod/2, 2] -->batch, len, d_mod/2, 2
		wk_phi_1_rev=wk_phi_1*torch.tensor([-1., 1.]).to(self.device)[None, None, None, ...] # (batch, len, d_mod/2, 2) * (1, 1, 1, 2)
		wk_phi_2 = torch.flip(wk_phi_1, dims = [-1]) ##[d_mod/2, 2] --># (batch, len, d_mod/2, 2)
	
		wk_phi1_2 = torch.cat((wk_phi_2, wk_phi_1_rev), axis = -1) #[dmod/2, 4] # (batch, len, d_mod/2, 4)
		wk_phi1_2_rehsaped = torch.reshape(wk_phi1_2, [batch*length*int(self.d_model/2), 2, 2]) #[dmod/2, 2, 2] --># (batch, len, d_mod/2, 2, 2) we want -->1*3*4*4

		transformation_matrix = torch.block_diag(*wk_phi1_2_rehsaped)
		out = torch.matmul(transformation_matrix, torch.reshape(inp, (batch*length*self.d_model, 1)))[:,0]
		out = torch.reshape(out, (length, self.d_model))
		return out

	def transform_by_delta_pos_v2(self, inp, delta_pos):
		#fast version, no need to use block diagonal matrix
		#transpose one token to another in the embedding space
		if inp.dim()==3:
			batch, length = int(inp.shape[0]), int(inp.shape[1])
		elif inp.dim()==1:
			batch, length = 1, int(inp.shape[0])

		raw = self.FMS(delta_pos)
		wk_phi_1 = torch.reshape(raw,[batch*length*int(self.d_model/2), 2]) #[d_mod/2, 2] -->batch* len* d_mod/2, 2
		wk_phi_1_rev=wk_phi_1*torch.tensor([-1., 1.]).to(self.device)[None, ...] # (batch*len*d_mod/2, 2) * (1, 2)
		wk_phi_2 = torch.flip(wk_phi_1, dims = [-1]) ##[d_mod/2, 2] --># (batch*len*d_mod/2, 2)
	
		wk_phi1_2 = torch.cat((wk_phi_2, wk_phi_1_rev), axis = -1) #[dmod/2, 4] # (batch* len* d_mod/2, 4)
		wk_phi1_2_rehsaped = torch.reshape(wk_phi1_2, [batch*length*int(self.d_model/2), 2, 2]) #[dmod/2, 2, 2] --># (batch* len*d_mod/2, 2, 2) we want -->1*3*4*4
		transformation_matrix = wk_phi1_2_rehsaped 
		
		if self.translation_bias is not None:
			inp -= self.translation_bias[:, None, :]

		reshaped = torch.reshape(inp, (batch*length*int(self.d_model/2), 2,1))
		out = torch.matmul(transformation_matrix, 
							reshaped) #(batch* len*d_mod/2, 2, 2) * (batch*len*d_mod, 1, 2)

		out = torch.reshape(out, (batch, length, self.d_model))
		if self.translation_bias is not None:
			out += self.translation_bias[:, None, :]
		return out


	def __call__(self, inp):
		if inp.dim()==2:
			inp = inp[..., None] #pos (batch, num_pitch, 1)
		elif inp.dim()==1:
			inp = inp[None, ..., None] #pos (1, num_pitch, 1)
		angle_rads = inp*self.angles #(batch, num_pitch)*(1,dim)

		# apply sin to even indices in the array; 2i
		angle_rads[:, :, 0::2] = torch.sin(angle_rads.clone()[:, : , 0::2])

		# apply cos to odd indices in the array; 2i+1
		angle_rads[:, :, 1::2] = torch.cos(angle_rads.clone()[:, :, 1::2])

		pos_encoding = angle_rads.to(torch.float32)
		if self.if_translation_bias:
			if self.translation_bias.size()[-1]!= self.d_model:
				translation_bias = self.translation_bias.repeat(1, 1,int(self.d_model/2))
			else:
				translation_bias = self.translation_bias
			pos_encoding += translation_bias
		else:
			self.translation_bias = None
		return pos_encoding
	
	def FMS(self, delta_pos):
		if delta_pos.dim()==1:
			delta_pos = delta_pos[None, ..., None] # len ==> batch, len
		if delta_pos.dim()==2:
			delta_pos = delta_pos[ ..., None] # batch, len ==> batch, len, 1
		if delta_pos.dim()==3:
			b_size = delta_pos.shape[0]
			len_q = delta_pos.shape[1]
			len_k = delta_pos.shape[2]
			delta_pos = delta_pos.reshape((b_size, len_q*len_k, 1))# batch, len, len ==> batch, len*len, 1
		
		raw = delta_pos*self.angles
		raw[:, :, 0::2] = torch.sin(raw.clone()[:, :, 0::2])
		raw[:,:,1::2] = torch.cos(raw.clone()[:,:,1::2])

		if delta_pos.dim()==3:
			raw = raw.reshape((b_size, len_q, len_k, -1))# batch, len, len ==> batch, len*len, 1
		return raw.to(torch.float32).to(self.device)

	def decode(self, embedded):
		if self.translation_bias is not None:
			embedded -= self.translation_bias[:, None, :]

		decoded_dim = (torch.asin(embedded)/self.angles[:, None, :]).to(torch.float32)
		if self.d_model/2 %2 == 0:
			decoded = decoded_dim[:, :, int(self.d_model/2)]

		elif self.d_model/2 %2 == 1:	
			decoded = decoded_dim[:, :, int(self.d_model/2+1)]

		return decoded 

	def decode_tps(self, embedded):
		decoded_dim = (torch.asin(embedded)/self.angles[:, None,None, :]).to(torch.float32)
		if self.d_model/2 %2 == 0:
			decoded = decoded_dim[:, :, :, int(self.d_model/2)]

		elif self.d_model/2 %2 == 1:	
			decoded = decoded_dim[:, :, :, int(self.d_model/2+1)]

		return decoded 

class Music_PositionalEncoding(nn.Module):

	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, if_index = True, if_global_timing = False, if_modulo_timing = False, device = 'cuda:0'):
		super().__init__()
		self.if_index = if_index
		self.if_global_timing = if_global_timing
		self.if_modulo_timing = if_modulo_timing
		self.dropout = nn.Dropout(p=dropout)
		self.index_embedding = Fundamental_Music_Embedding(d_model = d_model, base=10000, device = device, if_trainable=False, translation_bias_type = None, if_translation_bias_trainable = False, type = "se").cuda()
		self.global_time_embedding = Fundamental_Music_Embedding(d_model = d_model, base=10001, device = device, if_trainable=False, translation_bias_type = None, if_translation_bias_trainable = False, type = "se").cuda()
		self.modulo_time_embedding = Fundamental_Music_Embedding(d_model = d_model, base=10001, device = device, if_trainable=False, translation_bias_type = None, if_translation_bias_trainable = False, type = "se").cuda()

		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, 1, d_model)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)
		if self.if_global_timing:
			print("pe add global time")
		if self.if_modulo_timing:
			print("pe add modulo time")
		if self.if_index:
			print("pe add idx")
	def forward(self, inp,dur_onset_cumsum = None):

		if self.if_index:
			pe_index = self.pe[:inp.size(1)] #[seq_len, batch_size, embedding_dim]
			pe_index = torch.swapaxes(pe_index, 0, 1) #[batch_size, seq_len, embedding_dim]
			inp += pe_index
		
		if self.if_global_timing:
			global_timing = dur_onset_cumsum
			global_timing_embedding = self.global_time_embedding(global_timing)
			inp += global_timing_embedding
		
		if self.if_modulo_timing:
			modulo_timing = dur_onset_cumsum%4
			modulo_timing_embedding = self.modulo_time_embedding(modulo_timing)
			inp += modulo_timing_embedding
		return self.dropout(inp)
	
# class PositionalEncoding(nn.Module):

# 	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
# 		super().__init__()
# 		self.dropout = nn.Dropout(p=dropout)

# 		position = torch.arange(max_len).unsqueeze(1)
# 		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
# 		pe = torch.zeros(max_len, 1, d_model)
# 		pe[:, 0, 0::2] = torch.sin(position * div_term)
# 		pe[:, 0, 1::2] = torch.cos(position * div_term)
# 		self.register_buffer('pe', pe)

# 	def forward(self, x):
# 		pos = self.pe[:x.size(1)] #[seq_len, batch_size, embedding_dim]
# 		pos = torch.swapaxes(pos, 0, 1) #[batch_size, seq_len, embedding_dim]
# 		x = x + pos
# 		return self.dropout(x)

def l2_norm(a, b):
	return torch.linalg.norm(a-b,  ord = 2, dim = -1)

def rounding(x):
	return x-torch.sin(2.*math.pi*x)/(2.*math.pi)