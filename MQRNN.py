import torch
import torch.nn as nn
from Encoder import Encoder
from GNN_LSTM import GNN_LSTM
from Decoder import GlobalDecoder, LocalDecoder
from train import train
from data import MQRNN_dataset
from projection import *
from GNN_func import ConstraintGNN
from torch_geometric.data import Batch,Data
from Demand_Forecast_FC import DemandForecastFC


class MQRNN(nn.Module):
    """
    This class holds the encoder and the global decoder and local decoder.
    """

    def __init__(self,
                 horizon_size: int,
                 hidden_size: int,
                 quantiles: list,
                 columns: list,
                 dropout: float,
                 layer_size: int,
                 by_direction: bool,
                 lr: float,
                 batch_size: int,
                 num_epochs: int,
                 context_size: int,
                 covariate_size: int,
                 target_width: int,
                 x_width: int,
                 A_tensor,
                 B_tensor,
                 price,
                 device):
        #print(f"device is: {device}")
        super(MQRNN, self).__init__()
        self.device = device
        self.horizon_size = horizon_size
        self.hidden_size = hidden_size
        self.gnn_hidden = hidden_size / 4
        quantiles_size = len(quantiles)
        self.quantiles_size = quantiles_size
        self.quantiles = quantiles
        self.dropout = dropout
        self.layer_size = layer_size
        self.by_direction = by_direction
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.context_size = context_size
        self.covariate_size = covariate_size
        self.target_width = target_width
        self.x_shape = x_width
        self.A_tensor = A_tensor
        self.B_tensor = B_tensor
        self.price = price
        self.encoder = Encoder(horizon_size=horizon_size,
                               covariate_size=covariate_size,
                               hidden_size=hidden_size,
                               dropout=dropout,
                               layer_size=layer_size,
                               by_direction=by_direction,
                               target_width = target_width,
                               device=device)

        self.demand_forecast = DemandForecastFC(hidden_size=hidden_size,
                                          covariate_size=covariate_size,
                                          horizon_size=horizon_size,
                                      quantile_size=quantiles_size)

        self.gnn_lstm = GNN_LSTM(
                                x_shape=x_width,
                                horizon_size=horizon_size,
                               covariate_size=covariate_size,
                               hidden_size=hidden_size,
                               gnn_hidden=int(hidden_size / 4),
                               layer_size=layer_size,
                               target_width = target_width,
                               device=device)

        self.gdecoder = GlobalDecoder(hidden_size=hidden_size,
                                      covariate_size=covariate_size,
                                      horizon_size=horizon_size,
                                      context_size=context_size,
                                      target_width = target_width)
        self.ldecoder = LocalDecoder(covariate_size=covariate_size,
                                     quantile_size=quantiles_size,
                                     context_size=context_size,
                                     quantiles=quantiles,
                                     target_width = target_width,
                                     horizon_size=horizon_size)
        self.cGNN = ConstraintGNN(target_width=target_width,
                                  batch_size=batch_size)
        self.encoder.double()
        self.gnn_lstm.double()
        self.demand_forecast.double()
        self.gdecoder.double()
        self.ldecoder.double()
    # def forward(self, cur_series_covariate_tensor : torch.Tensor,
    #                   next_covariate_tensor: torch.Tensor,):
    #     LSTM_output = self.encoder(cur_series_covariate_tensor)
    #     hidden_and_covariate = torch.cat([LSTM_output, next_covariate_tensor], dim=2)
    #     Gdecoder_output = self.gdecoder(hidden_and_covariate)#[seq_len, batch_size, (horizon_size+1)*context_size]
    #     seq_len = Gdecoder_output.shape[1]
    #     #print(f"Gdecoder_output.shape: {Gdecoder_output.shape}")
    #     Gdecoder_output = Gdecoder_output.view(seq_len,self.batch_size,self.horizon_size+1,self.context_size)
    #     horizon_agnostic_context = Gdecoder_output[:,:,-1,:]
    #     horizon_specific_context = Gdecoder_output[:,:,:-1,:]
    #     horizon_agnostic_context = horizon_agnostic_context.repeat(1,1,self.horizon_size,1)
    #     next_covariate_tensor = next_covariate_tensor.view(seq_len,self.batch_size,self.horizon_size,self.covariate_size)
    #     Ldecoder_input = torch.cat([horizon_specific_context, next_covariate_tensor], dim=3)
    #     # print(f"horizon_agnostic_context.shape: {horizon_agnostic_context.shape}")
    #     # print(f"Ldecoder_input.shape: {Ldecoder_input.shape}")
    #     horizon_agnostic_context = horizon_agnostic_context.permute(1,0,2,3)
    #     Ldecoder_input = torch.cat([horizon_agnostic_context, Ldecoder_input],dim=3)#[seq_len, batch_size, horizon_size, 2*context_size+covariate_size]
    #     Ldecoder_output = self.ldecoder(Ldecoder_input)
    #     return Ldecoder_output

    def train(self, dataset: MQRNN_dataset,GNN_dataset:list):
        train(encoder=self.encoder,
              gnn_lstm=self.gnn_lstm,
              demand_forecast = self.demand_forecast,
                 gdecoder=self.gdecoder,
                 ldecoder=self.ldecoder,
                 cGNN=self.cGNN,
                 dataset=dataset,
                 GNN_dataset=GNN_dataset,
                 lr=self.lr,
                 batch_size=self.batch_size,
                 num_epochs=self.num_epochs,
                 target_width=self.target_width,
                 A_tensor = self.A_tensor,
                 B_tensor = self.B_tensor,
                 price = self.price,
                 device=self.device)
        #print("training finished")

    def predict(self, train_target_df, train_covariate_df, test_covariate_df, GNN_dataset, target_width, col_name, A_tensor, B_tensor):

        input_target_tensor = torch.tensor(train_target_df.to_numpy()) #[seq_len, target_width]
        full_covariate = train_covariate_df.to_numpy()
        full_covariate_tensor = torch.tensor(full_covariate) #[seq_len, covariate_size]

        next_covariate = test_covariate_df.to_numpy() #[horizon_size, covariate_size]
        next_covariate = next_covariate.reshape(-1, self.horizon_size * self.covariate_size)
        next_covariate_tensor = torch.tensor(next_covariate)  # [1,horizon_size * covariate_size]

        input_target_tensor = input_target_tensor.double()
        next_covariate_tensor = next_covariate_tensor.double()
        full_covariate_tensor = full_covariate_tensor.double()

        

        input_target_tensor = input_target_tensor.to(self.device)
        full_covariate_tensor = full_covariate_tensor.to(self.device)
        next_covariate_tensor = next_covariate_tensor.to(self.device)
        A_tensor = A_tensor.to(self.device)
        B_tensor = B_tensor.to(self.device)

        with torch.no_grad():
            input_target_covariate_tensor = torch.cat([input_target_tensor, full_covariate_tensor], dim=1)
            input_target_covariate_tensor = torch.unsqueeze(input_target_covariate_tensor,
                                                            dim=0)  # [1, seq_len, target_width + covariate_size]
            input_target_covariate_tensor = input_target_covariate_tensor.permute(1, 0,
                                                                                  2)  # [seq_len, 1, target_width + covariate_size]
            #print(f"input_target_covariate_tensor shape: {input_target_covariate_tensor.shape}")
            # outputs = self.encoder(input_target_covariate_tensor)   [seq_len,1,hidden_size]
            print("input_target_covariate_tensor_size",input_target_covariate_tensor.size())
            print("GNN_dataset_size",len(GNN_dataset))
            outputs, _ = self.gnn_lstm(input_target_covariate_tensor,GNN_dataset)
            # GNN_dataset = Batch.from_data_list(GNN_dataset)
            # GNN_hidden = self.cGNN(GNN_dataset)
            # GNN_hidden = GNN_hidden.to('cuda')
            # GNN_hidden = torch.unsqueeze(GNN_hidden[-1], dim=0)
            # print("GNN_output_size:",GNN_hidden.size()) 
            hidden = torch.unsqueeze(outputs[-1], dim=0)  # [1,1,hidden_size]
            print("hidden_size",hidden.size())

            next_covariate_tensor = torch.unsqueeze(next_covariate_tensor, dim=0) # [1,1, covariate_size * horizon_size]
            # next_covariate_tensor = torch.unsqueeze(next_covariate_tensor, dim=0) 

            #print(f"hidden shape: {hidden.shape}")
            #print(f"next_covariate_tensor: {next_covariate_tensor.shape}")
            gdecoder_input = torch.cat([hidden, next_covariate_tensor],
                                       dim=2)  # [1,1, hidden_size + covariate_size* horizon_size]
            gdecoder_output = self.gdecoder(gdecoder_input)  # [1,1,(horizon_size+1)*context_size]

            seq_len = gdecoder_output.shape[0]
            gdecoder_output = gdecoder_output.view(seq_len, self.batch_size, self.horizon_size + 1, self.context_size)
            demand_output = self.demand_forecast(gdecoder_input)
            print("demand_output.shape:",demand_output.shape)
            demand_output = demand_output.view(self.horizon_size, self.quantiles_size, 1)
            horizon_agnostic_context = gdecoder_output[:, :, -1, :]
            horizon_specific_context = gdecoder_output[:, :, :-1, :]
            horizon_agnostic_context = horizon_agnostic_context.repeat(1, 1, self.horizon_size, 1)
            next_covariate_tensor = next_covariate_tensor.view(seq_len, self.batch_size, self.horizon_size,
                                                               self.covariate_size)
            local_decoder_input = torch.cat([horizon_specific_context, next_covariate_tensor], dim=3)
            local_decoder_input = torch.cat([horizon_agnostic_context, local_decoder_input], dim=3)#[seq_len, batch_size, horizon_size, 2*context_size+covariate_size]
            #print(f"local_decoder_input shape: {local_decoder_input.shape}")
            local_decoder_output = self.ldecoder(
                local_decoder_input)  # [seq_len, batch_size, horizon_size* quantile_size, target_width]
            #print("local_decoder_output.shape:",local_decoder_output.shape)
            #print(local_decoder_output)
            local_decoder_output = local_decoder_output.view(self.horizon_size, self.quantiles_size, self.target_width*(self.target_width-1))
            local_decoder_output = projection_k(A_tensor, B_tensor, local_decoder_output)
            #print("local_decoder_output.shape:",local_decoder_output.shape)
            #return 0
            output_array = local_decoder_output.cpu().numpy()
            demand_output_array = demand_output.cpu().numpy()
            result_dict = {}
            demand = {}
            for i in range(self.quantiles_size):
                result_dict[self.quantiles[i]] = output_array[:, i,:]
                demand[self.quantiles[i]] = demand_output_array[:,i]
            return result_dict,demand