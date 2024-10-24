import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=5000):
		super(PositionalEncoding, self).__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		return x + self.pe[:x.size(0), :]


class TransformerPoseEstimation(nn.Module):
	def __init__(self, input_dim=1260, output_dim=54, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
		super(TransformerPoseEstimation, self).__init__()
		
		# Proyección inicial: adaptamos para que la entrada tenga el tamaño adecuado
		self.input_proj = nn.Linear(input_dim, d_model)
		
		# Codificación posicional
		self.positional_encoding = PositionalEncoding(d_model)
		
		# Encoder Transformer
		encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
		
		# Proyección final para las coordenadas en 3D
		self.output_proj = nn.Linear(d_model, output_dim)
		
		# Normalización y Dropout
		self.dropout = nn.Dropout(dropout)
		self.norm = nn.LayerNorm(d_model)


	def forward(self, x):
		# Suponemos que 'x' tiene forma (batch_size, input_dim) -> (batch_size, 1260)
		
		# Proyección de las características iniciales a un espacio d_model
		x = self.input_proj(x)  # (batch_size, d_model)
		
		# Añadir una dimensión de secuencia para que sea compatible con el transformer: (batch_size, 1, d_model)
		x = x.unsqueeze(1)
		
		# Transposición para que sea compatible con el Transformer: (1, batch_size, d_model)
		x = x.transpose(0, 1)
		
		# Añadir codificación posicional
		x = self.positional_encoding(x)
		
		# Aplicar el transformer encoder
		x = self.transformer_encoder(x)
		
		# Volver a transponer para obtener la forma original: (batch_size, 1, d_model)
		x = x.transpose(0, 1)
		
		# Quitar la dimensión de secuencia
		x = x.squeeze(1)
		
		# Proyectar las salidas del transformer a las coordenadas 3D
		x = self.output_proj(x)  # (batch_size, output_dim) -> (batch_size, 54)
		
		return x
