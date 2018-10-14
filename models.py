from modules import *
from torch.nn.functional import relu, tanh


class TextRNN(Module):
    """
    This  simple architecture  trains all data by using only the body of the post. It is a simple RNN
    where the final representation of a text is the output from the RNN of the last word. This last
    state passes from a MLP  giving as final output a scalar. In training, this output passes from a
    sigmoid in order to be transformed to probability.
    """
    def __init__(self, embeddings,
                 embeddings_dropout=0.0,
                 trainable_embeddings=True,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param trainable_embeddings: boolean indicating if the embedding will be trainable or frozen
        :param is_gru: GRU cell type if true, otherwise LSTM
        :param cell_hidden_size: the cell size of the RNN
        :param stacked_layers: the number of stacked layers of the RNN
        :param bidirectional: boolean indicating if the cell is bidirectional
        :param top_mlp_layers: number of layers of the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(TextRNN, self).__init__()
        self.input_list = ['text']
        self.name = "TextRNN"

        self.embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=trainable_embeddings)

        self.cell = CellLayer(is_gru, self.embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.last_state = LastState(self.cell.get_output_size(), self.decision_layer.get_input_size())
        self.seq = SequentialModel([self.embedding_layer, self.cell, self.last_state, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        out = self.seq(x)
        return out


class TextAuthorRNN(Module):
    """
    This model  trains all data by using the body and the author of the post. It is a simple RNN  but
    instead of using only the embedding of a word,  is used the concatenation of the embedding of the
    word with the embedding of the author of the post.
    """
    def __init__(self, embeddings,
                 author_embeddings_input_size,
                 embeddings_dropout=0.0,
                 trainable_embeddings=True,
                 author_embeddings_output_size=3,
                 author_embeddings_dropout=0.0,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):

        """
        :param embeddings: the matrix of the embeddings
        :param author_embeddings_input_size: the number of author embeddings that will be trained
        :param embeddings_dropout: dropout of the embeddings layer
        :param trainable_embeddings: boolean indicating if the embedding will be trainable or frozen
        :param author_embeddings_output_size:  the size of author embeddings
        :param author_embeddings_dropout: the dropout of the author embeddings
        :param is_gru: GRU cell type if true, otherwise LSTM
        :param cell_hidden_size: the cell size of the RNN
        :param stacked_layers: the number of stacked layers of the RNN
        :param bidirectional: boolean indicating if the cell is bidirectional
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(TextAuthorRNN, self).__init__()
        self.input_list = ['text', 'author']
        self.name = "TextAuthorRNN"
        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=trainable_embeddings)
        self.author_embedding_layer = EmbeddingLayer(input_size=author_embeddings_input_size,
                                                     output_size=author_embeddings_output_size, dropout=author_embeddings_dropout)
        self.cell = CellLayer(is_gru, self.word_embedding_layer.get_output_size() + self.author_embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.last_state = LastState(self.cell.get_output_size(), self.decision_layer.get_input_size())
        self.seq = SequentialModel([self.cell, self.last_state, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, y):
        x = self.word_embedding_layer(x)
        y = self.author_embedding_layer(y)
        u = y.unsqueeze(1).repeat(1, len(x[0]), 1)
        out = torch.cat((x, u), 2)
        out = self.seq(out)
        return out


class ProjectedTextRNN(Module):
    def __init__(self, embeddings,
                 embeddings_dropout=0.0,
                 proj_mlp_layers=1,
                 proj_mlp_activation=tanh,
                 proj_mlp_dropout=0.5,
                 proj_size=50,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param proj_mlp_layers: number of layers of the projection mlp
        :param proj_mlp_activation: activation function of the projection mlp (usually tanh)
        :param proj_mlp_dropout: dropout of the projection
        :param proj_size: hidden size of the projection layers
        :param is_gru: GRU cell type if true, otherwise LSTM
        :param cell_hidden_size: the cell size of the RNN
        :param stacked_layers: the number of stacked layers of the RNN
        :param bidirectional: boolean indicating if the cell is bidirectional
        :param top_mlp_layers: number of layers of the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(ProjectedTextRNN, self).__init__()
        self.input_list = ['text']
        self.name = "ProjectedTextRNN"

        self.embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=False)
        self.projection_layer = MultiLayerPerceptron(num_of_layers=proj_mlp_layers, init_size=self.embedding_layer.get_output_size(),
                                                     out_size=proj_size, dropout=proj_mlp_dropout,
                                                     inner_activation=proj_mlp_activation, outer_activation=proj_mlp_activation)

        self.cell = CellLayer(is_gru, self.embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.last_state = LastState(self.cell.get_output_size(), self.decision_layer.get_input_size())
        self.seq = SequentialModel([self.embedding_layer, self.projection_layer, self.cell, self.last_state, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        out = self.seq(x)
        return out


class ProjectedTextAuthorRNN(Module):
    def __init__(self, embeddings,
                 author_embeddings_input_size,
                 author_embeddings_output_size=3,
                 author_embeddings_dropout=0.0,
                 embeddings_dropout=0.0,
                 proj_mlp_layers=1,
                 proj_mlp_activation=tanh,
                 proj_mlp_dropout=0.5,
                 proj_size=50,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param author_embeddings_input_size: the number of author embeddings that will be trained
        :param author_embeddings_output_size:  the size of author embeddings
        :param author_embeddings_dropout: the dropout of the author embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param proj_mlp_layers: number of layers of the projection mlp
        :param proj_mlp_activation: activation function of the projection mlp (usually tanh)
        :param proj_mlp_dropout: dropout of the projection
        :param proj_size: hidden size of the projection layers
        :param is_gru: GRU cell type if true, otherwise LSTM
        :param cell_hidden_size: the cell size of the RNN
        :param stacked_layers: the number of stacked layers of the RNN
        :param bidirectional: boolean indicating if the cell is bidirectional
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(ProjectedTextAuthorRNN, self).__init__()
        self.input_list = ['text', 'author']
        self.name = "ProjectedTextAuthorRNN"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=False)
        self.projection_layer = MultiLayerPerceptron(num_of_layers=proj_mlp_layers, init_size=self.word_embedding_layer.get_output_size(),
                                                     out_size=proj_size, dropout=proj_mlp_dropout,
                                                     inner_activation=proj_mlp_activation, outer_activation=proj_mlp_activation)

        self.word_representation = SequentialModel([self.word_embedding_layer, self.projection_layer])
        self.author_embedding_layer = EmbeddingLayer(author_embeddings_input_size,
                                                     author_embeddings_output_size,
                                                     author_embeddings_dropout)

        self.cell = CellLayer(is_gru, self.word_embedding_layer.get_output_size() + self.author_embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.last_state = LastState(self.cell.get_output_size(), self.decision_layer.get_input_size())

        self.seq = SequentialModel([self.cell, self.last_state, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, y):
        x = self.word_embedding_layer(x)
        y = self.author_embedding_layer(y)
        u = y.unsqueeze(1).repeat(1, len(x[0]), 1)
        out = torch.cat((x, u), 2)
        out = self.seq(out)
        return out


class AttentionTextRNN(Module):
    def __init__(self, embeddings,
                 trainable_embeddings=True,
                 embeddings_dropout=0.0,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 att_mlp_layers=1,
                 att_mlp_dropout=0.5,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param trainable_embeddings: boolean indicating if the embedding will be trainable or frozen
        :param embeddings_dropout: dropout of the embeddings layer
        :param is_gru: GRU cell type if true, otherwise LSTM
        :param cell_hidden_size: the cell size of the RNN
        :param stacked_layers: the number of stacked layers of the RNN
        :param bidirectional: boolean indicating if the cell is bidirectional
        :param att_mlp_layers: number of layers of the attention mlp
        :param att_mlp_dropout: dropout of the attention mlp
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(AttentionTextRNN, self).__init__()
        self.input_list = ['text']
        self.name = "AttentionTextRNN"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=trainable_embeddings)

        self.cell = CellLayer(is_gru, self.word_embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        # self.last_state = LastState(self.cell.get_output_size(), self.decision_layer.get_input_size())
        self.last_state = AttendedState(att_mlp_layers, large_size, att_mlp_dropout, relu)
        self.seq = SequentialModel([self.word_embedding_layer, self.cell, self.last_state, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        out = self.seq(x)
        return out


class AttentionTextAuthorRNN(Module):
    def __init__(self, embeddings,
                 author_embeddings_input_size,
                 author_embeddings_output_size=3,
                 author_embeddings_dropout=0.0,
                 embeddings_dropout=0.0,
                 trainable_embeddings=True,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 att_mlp_layers=1,
                 att_mlp_dropout=0.5,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param author_embeddings_input_size: the number of author embeddings that will be trained
        :param author_embeddings_output_size:  the size of author embeddings
        :param author_embeddings_dropout: the dropout of the author embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param is_gru: GRU cell type if true, otherwise LSTM

        :param cell_hidden_size: the cell size of the RNN
        :param stacked_layers: the number of stacked layers of the RNN
        :param bidirectional: boolean indicating if the cell is bidirectional
        :param att_mlp_layers: number of layers of the attention mlp
        :param att_mlp_dropout: dropout of the attention mlp
        :param trainable_embeddings: boolean indicating if the embedding will be trainable or frozen
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(AttentionTextAuthorRNN, self).__init__()
        self.input_list = ['text', 'author']
        self.name = "AttentionTextAuthorRNN"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=trainable_embeddings)
        self.author_embedding_layer = EmbeddingLayer(author_embeddings_input_size,
                                                     author_embeddings_output_size,
                                                     author_embeddings_dropout)

        self.cell = CellLayer(is_gru, self.word_embedding_layer.get_output_size() + self.author_embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.last_state = AttendedState(att_mlp_layers, large_size, att_mlp_dropout, relu)
        self.seq = SequentialModel([self.cell, self.last_state, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, y):
        x = self.word_embedding_layer(x)
        y = self.author_embedding_layer(y)
        u = y.unsqueeze(1).repeat(1, len(x[0]), 1)
        out = torch.cat((x, u), 2)
        out = self.seq(out)
        return out


class ProjectedAttentionTextRNN(Module):
    def __init__(self, embeddings,
                 embeddings_dropout=0.0,
                 proj_mlp_layers=1,
                 proj_mlp_activation=tanh,
                 proj_mlp_dropout=0.5,
                 proj_size=50,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 att_mlp_layers=1,
                 att_mlp_dropout=0.5,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param proj_mlp_layers: number of layers of the projection mlp
        :param proj_mlp_activation: activation function of the projection mlp (usually tanh)
        :param proj_mlp_dropout: dropout of the projection
        :param proj_size: hidden size of the projection layers
        :param is_gru: GRU cell type if true, otherwise LSTM
        :param cell_hidden_size: the cell size of the RNN
        :param stacked_layers: the number of stacked layers of the RNN
        :param bidirectional: boolean indicating if the cell is bidirectional
        :param att_mlp_layers: number of layers of the attention mlp
        :param att_mlp_dropout: dropout of the attention mlp
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(ProjectedAttentionTextRNN, self).__init__()
        self.input_list = ['text']
        self.name = "ProjectedAttentionTextRNN"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=False)

        self.projection_layer = MultiLayerPerceptron(num_of_layers=proj_mlp_layers, init_size=self.word_embedding_layer.get_output_size(),
                                                     out_size=proj_size, dropout=proj_mlp_dropout,
                                                     inner_activation=proj_mlp_activation, outer_activation=proj_mlp_activation)

        self.cell = CellLayer(is_gru, self.word_embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.last_state = AttendedState(att_mlp_layers, large_size, att_mlp_dropout, relu)
        self.seq = SequentialModel([self.word_embedding_layer, self.projection_layer, self.cell, self.last_state, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        out = self.seq(x)
        return out


class ProjectedAttentionTextAuthorRNN(Module):
    def __init__(self, embeddings,
                 author_embeddings_input_size,
                 author_embeddings_output_size=3,
                 author_embeddings_dropout=0.0,
                 embeddings_dropout=0.0,
                 proj_mlp_layers=1,
                 proj_mlp_activation=tanh,
                 proj_mlp_dropout=0.5,
                 proj_size=50,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 att_mlp_layers=1,
                 att_mlp_dropout=0.5,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param author_embeddings_input_size: the number of author embeddings that will be trained
        :param author_embeddings_output_size:  the size of author embeddings
        :param author_embeddings_dropout: the dropout of the author embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param proj_mlp_layers: number of layers of the projection mlp
        :param proj_mlp_activation: activation function of the projection mlp (usually tanh)
        :param proj_mlp_dropout: dropout of the projection
        :param proj_size: hidden size of the projection layers
        :param is_gru: GRU cell type if true, otherwise LSTM
        :param cell_hidden_size: the cell size of the RNN
        :param stacked_layers: the number of stacked layers of the RNN
        :param bidirectional: boolean indicating if the cell is bidirectional
        :param att_mlp_layers: number of layers of the attention mlp
        :param att_mlp_dropout: dropout of the attention mlp
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(ProjectedAttentionTextAuthorRNN, self).__init__()
        self.input_list = ['text', 'author']
        self.name = "ProjectedAttentionTextAuthorRNN"
        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=False)
        self.projection_layer = MultiLayerPerceptron(num_of_layers=proj_mlp_layers,
                                                     init_size=self.word_embedding_layer.get_output_size(),
                                                     out_size=proj_size, dropout=proj_mlp_dropout,
                                                     inner_activation=proj_mlp_activation, outer_activation=proj_mlp_activation)
        self.author_embedding_layer = EmbeddingLayer(author_embeddings_input_size,
                                                     author_embeddings_output_size,
                                                     author_embeddings_dropout)

        self.word_representation = SequentialModel([self.word_embedding_layer, self.projection_layer])

        self.cell = CellLayer(is_gru, self.word_representation.get_output_size() + self.author_embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        # self.last_state = LastState(self.cell.get_output_size(), self.decision_layer.get_input_size())
        self.last_state = AttendedState(att_mlp_layers, large_size, att_mlp_dropout, relu)
        self.seq = SequentialModel([self.cell, self.last_state, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, y):
        x = self.word_representation(x)
        y = self.author_embedding_layer(y)
        u = y.unsqueeze(1).repeat(1, len(x[0]), 1)
        out = torch.cat((x, u), 2)
        out = self.seq(out)
        return out


class TextTitleRNN(Module):
    def __init__(self, embeddings,
                 embeddings_dropout=0.0,
                 trainable_embeddings=True,
                 text_is_gru=True,
                 text_cell_hidden_size=128,
                 text_stacked_layers=1,
                 text_bidirectional=False,
                 title_is_gru=True,
                 title_cell_hidden_size=32,
                 title_stacked_layers=1,
                 title_bidirectional=False,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param trainable_embeddings: boolean indicating if the embedding will be trainable or frozen
        :param text_is_gru: GRU cell type for the text cell if true, otherwise LSTM
        :param text_cell_hidden_size: the cell size of the RNN for text representation
        :param text_stacked_layers: the number of stacked layers of the RNN for text representation
        :param text_bidirectional: boolean indicating if the RNN for text representation is bidirectional
        :param title_is_gru: GRU cell type for the title cell if true, otherwise LSTM
        :param title_cell_hidden_size: the cell size of the RNN for title representation
        :param title_stacked_layers: the number of stacked layers of the RNN for title representation
        :param title_bidirectional: boolean indicating if the RNN for title representation is bidirectional
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(TextTitleRNN, self).__init__()
        self.input_list = ['text', 'title']
        self.name = "TextTitleRNN"
        self.embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=trainable_embeddings)
        self.text_rnn = CellLayer(text_is_gru, self.embedding_layer.get_output_size(),
                                  text_cell_hidden_size, text_bidirectional, text_stacked_layers)
        self.title_rnn = CellLayer(title_is_gru, self.embedding_layer.get_output_size(),
                                   title_cell_hidden_size, title_bidirectional, title_stacked_layers)
        large_size1 = text_cell_hidden_size * 2 if text_bidirectional else text_cell_hidden_size
        large_size2 = title_cell_hidden_size * 2 if title_bidirectional else title_cell_hidden_size

        self.text_state = LastState(large_size1, large_size1)
        self.title_state = LastState(large_size2, large_size2)
        self.text_representation = SequentialModel([self.embedding_layer, self.text_rnn, self.text_state])
        self.title_representation = SequentialModel([self.embedding_layer, self.title_rnn, self.title_state])
        self.concatenation = ConcatenationLayer(self.text_representation, self.title_representation)
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size1 + large_size2,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        # self.seq = SequentialModel([self.concatenation, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, y):
        x = self.text_representation(x)
        y = self.title_representation(y)
        output = self.concatenation(x, y)
        output = self.decision_layer(output)
        return output


class TextTitleAuthorRNN(Module):
    def __init__(self, embeddings,
                 author_embeddings_input_size,
                 author_embeddings_output_size=3,
                 author_embeddings_dropout=0.0,
                 embeddings_dropout=0.0,
                 trainable_embeddings=True,
                 text_is_gru=True,
                 text_cell_hidden_size=128,
                 text_stacked_layers=1,
                 text_bidirectional=False,
                 title_is_gru=True,
                 title_cell_hidden_size=32,
                 title_stacked_layers=1,
                 title_bidirectional=False,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param author_embeddings_input_size: the number of author embeddings that will be trained
        :param author_embeddings_output_size:  the size of author embeddings
        :param author_embeddings_dropout: the dropout of the author embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param trainable_embeddings: boolean indicating if the embedding will be trainable or frozen
        :param text_is_gru: GRU cell type for the text cell if true, otherwise LSTM
        :param text_cell_hidden_size: the cell size of the RNN for text representation
        :param text_stacked_layers: the number of stacked layers of the RNN for text representation
        :param text_bidirectional: boolean indicating if the RNN for text representation is bidirectional
        :param title_is_gru: GRU cell type for the title cell if true, otherwise LSTM
        :param title_cell_hidden_size: the cell size of the RNN for title representation
        :param title_stacked_layers: the number of stacked layers of the RNN for title representation
        :param title_bidirectional: boolean indicating if the RNN for title representation is bidirectional
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(TextTitleAuthorRNN, self).__init__()
        self.input_list = ['text', 'title', 'author']
        self.name = "TextTitleAuthorRNN"
        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=trainable_embeddings)
        self.author_embedding_layer = EmbeddingLayer(input_size=author_embeddings_input_size,
                                                     output_size=author_embeddings_output_size, dropout=author_embeddings_dropout)
        self.text_rnn = CellLayer(text_is_gru, self.word_embedding_layer.get_output_size() + self.author_embedding_layer.get_output_size(),
                                  text_cell_hidden_size, text_bidirectional, text_stacked_layers)
        self.title_rnn = CellLayer(title_is_gru, self.word_embedding_layer.get_output_size() + self.author_embedding_layer.get_output_size(),
                                   title_cell_hidden_size, title_bidirectional, title_stacked_layers)
        large_size1 = text_cell_hidden_size * 2 if text_bidirectional else text_cell_hidden_size
        large_size2 = title_cell_hidden_size * 2 if title_bidirectional else title_cell_hidden_size

        self.text_state = LastState(large_size1, large_size1)
        self.title_state = LastState(large_size2, large_size2)
        self.text_representation = SequentialModel([self.text_rnn, self.text_state])
        self.title_representation = SequentialModel([self.title_rnn, self.title_state])
        self.concatenation = ConcatenationLayer(self.text_representation, self.title_representation)
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size1 + large_size2,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        # self.seq = SequentialModel([self.concatenation, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, y, u):
        x = self.word_embedding_layer(x)
        y = self.word_embedding_layer(y)
        u = self.author_embedding_layer(u)
        u_x = u.unsqueeze(1).repeat(1, len(x[0]), 1)
        u_y = u.unsqueeze(1).repeat(1, len(y[0]), 1)
        x = torch.cat((x, u_x), 2)
        y = torch.cat((y, u_y), 2)
        x = self.text_representation(x)
        y = self.title_representation(y)

        output = self.concatenation(x, y)
        output = self.decision_layer(output)
        return output


class AttentionTextTitleRNN(Module):
    def __init__(self, embeddings,
                 embeddings_dropout=0.0,
                 trainable_embeddings=True,
                 text_is_gru=True,
                 text_cell_hidden_size=128,
                 text_stacked_layers=1,
                 text_bidirectional=False,
                 title_is_gru=True,
                 title_cell_hidden_size=32,
                 title_stacked_layers=1,
                 title_bidirectional=False,
                 text_att_mlp_layers=0,
                 text_att_mlp_activation=relu,
                 text_att_dropout=0.0,
                 title_att_mlp_layers=0,
                 title_att_mlp_activation=relu,
                 title_att_dropout=0.0,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):

        """
        :param embeddings: the matrix of the embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param trainable_embeddings: boolean indicating if the embedding will be trainable or frozen
        :param text_is_gru: GRU cell type for the text cell if true, otherwise LSTM
        :param text_cell_hidden_size: the cell size of the RNN for text representation
        :param text_bidirectional: boolean indicating if the RNN for text representation is bidirectional
        :param title_is_gru: GRU cell type for the title cell if true, otherwise LSTM
        :param title_cell_hidden_size: the cell size of the RNN for title representation
        :param title_stacked_layers: the number of stacked layers of the RNN for title representation
        :param title_bidirectional: boolean indicating if the RNN for title representation is bidirectional
        :param top_mlp_layers: number of layers for the top mlp
        :param text_att_mlp_layers: number of layers of the text attention mlp
        :param text_att_mlp_activation: activation of the text attention mlp
        :param text_att_dropout: dropout of the text attention mlp
        :param title_att_mlp_layers: number of layers of the title attention mlp
        :param title_att_mlp_activation: activation of the title attention mlp
        :param title_att_dropout: dropout of the title attention mlp
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(AttentionTextTitleRNN, self).__init__()
        self.input_list = ['text', 'title']
        self.name = "AttentionTextTitleRNN"
        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=trainable_embeddings)

        self.text_rnn = CellLayer(text_is_gru, self.word_embedding_layer.get_output_size(),
                                  text_cell_hidden_size, text_bidirectional, text_stacked_layers)
        self.title_rnn = CellLayer(title_is_gru, self.word_embedding_layer.get_output_size(),
                                   title_cell_hidden_size, title_bidirectional, title_stacked_layers)
        large_size1 = text_cell_hidden_size * 2 if text_bidirectional else text_cell_hidden_size
        large_size2 = title_cell_hidden_size * 2 if title_bidirectional else title_cell_hidden_size

        self.text_state = AttendedState(text_att_mlp_layers, large_size1, text_att_dropout, text_att_mlp_activation)
        self.title_state = AttendedState(title_att_mlp_layers, large_size2, title_att_dropout, title_att_mlp_activation)

        self.text_representation = SequentialModel([self.word_embedding_layer, self.text_rnn, self.text_state])
        self.title_representation = SequentialModel([self.word_embedding_layer, self.title_rnn, self.title_state])
        self.concatenation = ConcatenationLayer(self.text_representation, self.title_representation)
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size1 + large_size2,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, y):
        x = self.text_representation(x)
        y = self.title_representation(y)
        output = self.concatenation(x, y)
        output = self.decision_layer(output)
        return output


class AttentionTextTitleAuthorRNN(Module):
    def __init__(self, embeddings,
                 author_embeddings_input_size,
                 embeddings_dropout=0.0,
                 trainable_embeddings=True,
                 author_embeddings_output_size=3,
                 author_embeddings_dropout=0.0,
                 text_is_gru=True,
                 text_cell_hidden_size=128,
                 text_stacked_layers=1,
                 text_bidirectional=False,
                 title_is_gru=True,
                 title_cell_hidden_size=32,
                 title_stacked_layers=1,
                 title_bidirectional=False,
                 text_att_mlp_layers=0,
                 text_att_mlp_activation=relu,
                 text_att_dropout=0.0,
                 title_att_mlp_layers=0,
                 title_att_mlp_activation=relu,
                 title_att_dropout=0.0,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):

        """
        :param embeddings: the matrix of the embeddings
        :param author_embeddings_input_size: the number of author embeddings that will be trained
        :param author_embeddings_output_size:  the size of author embeddings
        :param author_embeddings_dropout: the dropout of the author embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param trainable_embeddings: boolean indicating if the embedding will be trainable or frozen
        :param text_is_gru: GRU cell type for the text cell if true, otherwise LSTM
        :param text_cell_hidden_size: the cell size of the first RNN
        :param text_stacked_layers: the number of stacked layers of the first RNN
        :param text_bidirectional: boolean indicating if the first cell is bidirectional
        :param title_is_gru: GRU cell type for the title cell if true, otherwise LSTM
        :param title_cell_hidden_size: the cell size of the RNN for title representation
        :param title_stacked_layers: the number of stacked layers of the RNN for title representation
        :param title_bidirectional: boolean indicating if the RNN for title representation is bidirectional
        :param text_att_mlp_layers: number of layers of the text attention mlp
        :param text_att_mlp_activation: activation of the text attention mlp
        :param text_att_dropout: dropout of the text attention mlp
        :param title_att_mlp_layers: number of layers of the title attention mlp
        :param title_att_mlp_activation: activation of the title attention mlp
        :param title_att_dropout: dropout of the title attention mlp
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(AttentionTextTitleAuthorRNN, self).__init__()
        self.input_list = ['text', 'title', 'author']
        self.name = "AttentionTextTitleAuthorRNN"
        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=trainable_embeddings)
        self.author_embedding_layer = EmbeddingLayer(input_size=author_embeddings_input_size,
                                                     output_size=author_embeddings_output_size, dropout=author_embeddings_dropout)
        self.text_rnn = CellLayer(text_is_gru, self.word_embedding_layer.get_output_size() + self.author_embedding_layer.get_output_size(),
                                  text_cell_hidden_size, text_bidirectional, text_stacked_layers)
        self.title_rnn = CellLayer(title_is_gru, self.word_embedding_layer.get_output_size() + self.author_embedding_layer.get_output_size(),
                                   title_cell_hidden_size, title_bidirectional, title_stacked_layers)
        large_size1 = text_cell_hidden_size * 2 if text_bidirectional else text_cell_hidden_size
        large_size2 = title_cell_hidden_size * 2 if title_bidirectional else title_cell_hidden_size

        self.text_state = AttendedState(text_att_mlp_layers, large_size1, text_att_dropout, text_att_mlp_activation)
        self.title_state = AttendedState(title_att_mlp_layers, large_size2, title_att_dropout, title_att_mlp_activation)

        self.text_representation = SequentialModel([self.text_rnn, self.text_state])
        self.title_representation = SequentialModel([self.title_rnn, self.title_state])
        self.concatenation = ConcatenationLayer(self.text_representation, self.title_representation)
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size1 + large_size2,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, y, u):
        x = self.word_embedding_layer(x)
        y = self.word_embedding_layer(y)
        u = self.author_embedding_layer(u)
        u_x = u.unsqueeze(1).repeat(1, len(x[0]), 1)
        u_y = u.unsqueeze(1).repeat(1, len(y[0]), 1)
        x = torch.cat((x, u_x), 2)
        y = torch.cat((y, u_y), 2)
        x = self.text_representation(x)
        y = self.title_representation(y)
        output = self.concatenation(x, y)
        output = self.decision_layer(output)
        return output


class ProjectedTextTitleRNN(Module):
    def __init__(self, embeddings,
                 embeddings_dropout=0.0,
                 proj_mlp_dropout=0.0,
                 proj_mlp_layers=1,
                 proj_mlp_activation=tanh,
                 proj_hidden_size=100,
                 text_is_gru=True,
                 text_cell_hidden_size=128,
                 text_stacked_layers=1,
                 text_bidirectional=False,
                 title_is_gru=True,
                 title_cell_hidden_size=32,
                 title_stacked_layers=1,
                 title_bidirectional=False,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param proj_mlp_layers: number of layers of the projection mlp
        :param proj_mlp_activation: activation function of the projection mlp (usually tanh)
        :param proj_mlp_dropout: dropout of the projection
        :param proj_hidden_size: hidden size of the projection layers
        :param text_is_gru: GRU cell type for the text cell if true, otherwise LSTM
        :param text_cell_hidden_size: the cell size of the first RNN
        :param text_stacked_layers: the number of stacked layers of the first RNN
        :param text_bidirectional: boolean indicating if the first cell is bidirectional
        :param title_is_gru: GRU cell type for the title cell if true, otherwise LSTM
        :param title_cell_hidden_size: the cell size of the RNN for title representation
        :param title_stacked_layers: the number of stacked layers of the RNN for title representation
        :param title_bidirectional: boolean indicating if the RNN for title representation is bidirectional
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(ProjectedTextTitleRNN, self).__init__()
        self.input_list = ['text', 'title']
        self.name = "ProjectedTextTitleRNN"
        self.embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=False)
        self.projection_layer = MultiLayerPerceptron(num_of_layers=proj_mlp_layers,
                                                     dropout=proj_mlp_dropout,
                                                     init_size=self.embedding_layer.get_output_size(),
                                                     out_size=proj_hidden_size,
                                                     inner_activation=proj_mlp_activation,
                                                     outer_activation=proj_mlp_activation
                                                     )
        self.text_rnn = CellLayer(text_is_gru, proj_hidden_size,
                                  text_cell_hidden_size, text_bidirectional, text_stacked_layers)
        self.title_rnn = CellLayer(title_is_gru, proj_hidden_size,
                                   title_cell_hidden_size, title_bidirectional, title_stacked_layers)
        large_size1 = text_cell_hidden_size * 2 if text_bidirectional else text_cell_hidden_size
        large_size2 = title_cell_hidden_size * 2 if title_bidirectional else title_cell_hidden_size

        self.text_state = LastState(large_size1, large_size1)
        self.title_state = LastState(large_size2, large_size2)

        self.text_representation = SequentialModel([self.embedding_layer,self.projection_layer, self.text_rnn, self.text_state])
        self.title_representation = SequentialModel([self.embedding_layer, self.projection_layer, self.title_rnn, self.title_state])
        self.concatenation = ConcatenationLayer(self.text_representation, self.title_representation)
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size1 + large_size2,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, y):
        x = self.text_representation(x)
        y = self.title_representation(y)
        output = self.concatenation(x, y)
        output = self.decision_layer(output)
        return output


class ProjectedTextTitleAuthorRNN(Module):
    def __init__(self, embeddings,
                 author_embeddings_input_size,
                 embeddings_dropout=0.0,
                 author_embeddings_output_size=3,
                 author_embeddings_dropout=0.0,
                 proj_mlp_dropout=0.0,
                 proj_mlp_layers=1,
                 proj_mlp_activation=tanh,
                 proj_hidden_size=100,
                 text_is_gru=True,
                 text_cell_hidden_size=128,
                 text_stacked_layers=1,
                 text_bidirectional=False,
                 title_is_gru=True,
                 title_cell_hidden_size=32,
                 title_stacked_layers=1,
                 title_bidirectional=False,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param author_embeddings_input_size: the number of author embeddings that will be trained
        :param author_embeddings_output_size:  the size of author embeddings
        :param author_embeddings_dropout: the dropout of the author embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param proj_mlp_layers: number of layers of the projection mlp
        :param proj_mlp_activation: activation function of the projection mlp (usually tanh)
        :param proj_mlp_dropout: dropout of the projection
        :param proj_hidden_size: hidden size of the projection layers
        :param text_is_gru: GRU cell type for the text cell if true, otherwise LSTM
        :param text_cell_hidden_size: the cell size of the first RNN
        :param text_stacked_layers: the number of stacked layers of the first RNN
        :param text_bidirectional: boolean indicating if the first cell is bidirectional
        :param title_is_gru: GRU cell type for the title cell if true, otherwise LSTM
        :param title_cell_hidden_size: the cell size of the RNN for title representation
        :param title_stacked_layers: the number of stacked layers of the RNN for title representation
        :param title_bidirectional: boolean indicating if the RNN for title representation is bidirectional
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(ProjectedTextTitleAuthorRNN, self).__init__()
        self.input_list = ['text', 'title', 'author']
        self.name = "ProjectedTextTitleAuthorRNN"
        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=False)

        self.projection_layer = MultiLayerPerceptron(num_of_layers=proj_mlp_layers,
                                                     dropout=proj_mlp_dropout,
                                                     init_size=self.word_embedding_layer.get_output_size(),
                                                     out_size=proj_hidden_size,
                                                     inner_activation=proj_mlp_activation,
                                                     outer_activation=proj_mlp_activation)

        self.word_representation = SequentialModel([self.word_embedding_layer, self.projection_layer])

        self.author_embedding_layer = EmbeddingLayer(input_size=author_embeddings_input_size,
                                                     output_size=author_embeddings_output_size, dropout=author_embeddings_dropout)
        self.text_rnn = CellLayer(text_is_gru, self.word_representation.get_output_size() + self.author_embedding_layer.get_output_size(),
                                  text_cell_hidden_size, text_bidirectional, text_stacked_layers)
        self.title_rnn = CellLayer(title_is_gru, self.word_representation.get_output_size() + self.author_embedding_layer.get_output_size(),
                                   title_cell_hidden_size, title_bidirectional, title_stacked_layers)
        large_size1 = text_cell_hidden_size * 2 if text_bidirectional else text_cell_hidden_size
        large_size2 = title_cell_hidden_size * 2 if title_bidirectional else title_cell_hidden_size

        self.text_state = LastState(large_size1, large_size1)
        self.title_state = LastState(large_size2, large_size2)

        self.text_representation = SequentialModel([self.text_rnn, self.text_state])
        self.title_representation = SequentialModel([self.title_rnn, self.title_state])
        self.concatenation = ConcatenationLayer(self.text_representation, self.title_representation)
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size1 + large_size2,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, y, u):
        x = self.word_representation(x)
        y = self.word_representation(y)
        u = self.author_embedding_layer(u)
        u_x = u.unsqueeze(1).repeat(1, len(x[0]), 1)
        u_y = u.unsqueeze(1).repeat(1, len(y[0]), 1)
        x = torch.cat((x, u_x), 2)
        y = torch.cat((y, u_y), 2)
        x = self.text_representation(x)
        y = self.title_representation(y)
        output = self.concatenation(x, y)
        output = self.decision_layer(output)
        return output


class ProjectedAttentionTextTitleRNN(Module):
    def __init__(self, embeddings,
                 embeddings_dropout=0.0,
                 proj_mlp_dropout=0.0,
                 proj_mlp_layers=1,
                 proj_mlp_activation=tanh,
                 proj_hidden_size=100,
                 text_is_gru=True,
                 text_cell_hidden_size=128,
                 text_stacked_layers=1,
                 text_bidirectional=False,
                 title_is_gru=True,
                 title_cell_hidden_size=32,
                 title_stacked_layers=1,
                 title_bidirectional=False,
                 text_att_mlp_layers=0,
                 text_att_mlp_activation=relu,
                 text_att_dropout=0.0,
                 title_att_mlp_layers=0,
                 title_att_mlp_activation=relu,
                 title_att_dropout=0.0,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param proj_mlp_layers: number of layers of the projection mlp
        :param proj_mlp_activation: activation function of the projection mlp (usually tanh)
        :param proj_mlp_dropout: dropout of the projection
        :param proj_hidden_size: hidden size of the projection layers
        :param text_is_gru: GRU cell type for the text cell if true, otherwise LSTM
        :param text_cell_hidden_size: the cell size of the first RNN
        :param text_stacked_layers: the number of stacked layers of the first RNN
        :param text_bidirectional: boolean indicating if the first cell is bidirectional
        :param title_is_gru: GRU cell type for the title cell if true, otherwise LSTM
        :param title_cell_hidden_size: the cell size of the RNN for title representation
        :param title_stacked_layers: the number of stacked layers of the RNN for title representation
        :param title_bidirectional: boolean indicating if the RNN for title representation is bidirectional
        :param text_att_mlp_layers: number of layers of the text attention mlp
        :param text_att_mlp_activation: activation of the text attention mlp
        :param text_att_dropout: dropout of the text attention mlp
        :param title_att_mlp_layers: number of layers of the title attention mlp
        :param title_att_mlp_activation: activation of the title attention mlp
        :param title_att_dropout: dropout of the title attention mlp
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(ProjectedAttentionTextTitleRNN, self).__init__()
        self.input_list = ['text', 'title']
        self.name = "ProjectedAttentionTextTitleRNN"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=False)
        self.projection_layer = MultiLayerPerceptron(num_of_layers=proj_mlp_layers,
                                                     dropout=proj_mlp_dropout,
                                                     init_size=self.word_embedding_layer.get_output_size(),
                                                     out_size=proj_hidden_size,
                                                     inner_activation=proj_mlp_activation,
                                                     outer_activation=proj_mlp_activation)

        self.text_rnn = CellLayer(text_is_gru, self.word_representation.get_output_size(),
                                  text_cell_hidden_size, text_bidirectional, text_stacked_layers)
        self.title_rnn = CellLayer(title_is_gru, self.word_representation.get_output_size(),
                                   title_cell_hidden_size, title_bidirectional, title_stacked_layers)
        large_size1 = text_cell_hidden_size * 2 if text_bidirectional else text_cell_hidden_size
        large_size2 = title_cell_hidden_size * 2 if title_bidirectional else title_cell_hidden_size

        self.text_state = AttendedState(text_att_mlp_layers, large_size1, text_att_dropout, text_att_mlp_activation)
        self.title_state = AttendedState(title_att_mlp_layers, large_size2, title_att_dropout, title_att_mlp_activation)
        self.text_representation = SequentialModel([self.word_embedding_layer, self.projection_layer, self.text_rnn, self.text_state])
        self.title_representation = SequentialModel([self.word_embedding_layer, self.projection_layer, self.title_rnn, self.title_state])
        self.concatenation = ConcatenationLayer(self.text_representation, self.title_representation)
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size1 + large_size2,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, y):
        x = self.text_representation(x)
        y = self.title_representation(y)
        output = self.concatenation(x, y)
        output = self.decision_layer(output)
        return output


class ProjectedAttentionTextTitleAuthorRNN(Module):
    def __init__(self, embeddings,
                 author_embeddings_input_size,
                 author_embeddings_output_size=3,
                 author_embeddings_dropout=0.0,
                 embeddings_dropout=0.0,
                 proj_mlp_dropout=0.0,
                 proj_mlp_layers=1,
                 proj_mlp_activation=tanh,
                 proj_hidden_size=100,
                 text_is_gru=True,
                 text_cell_hidden_size=128,
                 text_stacked_layers=1,
                 text_bidirectional=False,
                 title_is_gru=True,
                 title_cell_hidden_size=32,
                 title_stacked_layers=1,
                 title_bidirectional=False,
                 text_att_mlp_layers=0,
                 text_att_mlp_activation=relu,
                 text_att_dropout=0.0,
                 title_att_mlp_layers=0,
                 title_att_mlp_activation=relu,
                 title_att_dropout=0.0,
                 top_mlp_dropout=0.0,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None):
        """
        :param embeddings: the matrix of the embeddings
        :param author_embeddings_input_size: the number of author embeddings that will be trained
        :param author_embeddings_output_size:  the size of author embeddings
        :param author_embeddings_dropout: the dropout of the author embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param proj_mlp_layers: number of layers of the projection mlp
        :param proj_mlp_activation: activation function of the projection mlp (usually tanh)
        :param proj_mlp_dropout: dropout of the projection
        :param proj_hidden_size: hidden size of the projection layers
        :param text_is_gru: GRU cell type for the text cell if true, otherwise LSTM
        :param text_cell_hidden_size: the cell size of the first RNN
        :param text_stacked_layers: the number of stacked layers of the first RNN
        :param text_bidirectional: boolean indicating if the first cell is bidirectional
        :param title_is_gru: GRU cell type for the title cell if true, otherwise LSTM
        :param title_cell_hidden_size: the cell size of the RNN for title representation
        :param title_stacked_layers: the number of stacked layers of the RNN for title representation
        :param title_bidirectional: boolean indicating if the RNN for title representation is bidirectional
        :param text_att_mlp_layers: number of layers of the text attention mlp
        :param text_att_mlp_activation: activation of the text attention mlp
        :param text_att_dropout: dropout of the text attention mlp
        :param title_att_mlp_layers: number of layers of the title attention mlp
        :param title_att_mlp_activation: activation of the title attention mlp
        :param title_att_dropout: dropout of the title attention mlp
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(ProjectedAttentionTextTitleAuthorRNN, self).__init__()
        self.input_list = ['text', 'title', 'author']
        self.name = "ProjectedAttentionTextTitleAuthorRNN"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=False)
        self.projection_layer = MultiLayerPerceptron(num_of_layers=proj_mlp_layers,
                                                     dropout=proj_mlp_dropout,
                                                     init_size=self.word_embedding_layer.get_output_size(),
                                                     out_size=proj_hidden_size,
                                                     inner_activation=proj_mlp_activation,
                                                     outer_activation=proj_mlp_activation)

        self.word_representation = SequentialModel([self.word_embedding_layer, self.projection_layer])

        self.author_embedding_layer = EmbeddingLayer(input_size=author_embeddings_input_size,
                                                     output_size=author_embeddings_output_size, dropout=author_embeddings_dropout)
        self.text_rnn = CellLayer(text_is_gru, self.word_representation.get_output_size() + self.author_embedding_layer.get_output_size(),
                                  text_cell_hidden_size, text_bidirectional, text_stacked_layers)
        self.title_rnn = CellLayer(title_is_gru, self.word_representation.get_output_size() + self.author_embedding_layer.get_output_size(),
                                   title_cell_hidden_size, title_bidirectional, title_stacked_layers)
        large_size1 = text_cell_hidden_size * 2 if text_bidirectional else text_cell_hidden_size
        large_size2 = title_cell_hidden_size * 2 if title_bidirectional else title_cell_hidden_size

        self.text_state = AttendedState(text_att_mlp_layers, large_size1, text_att_dropout, text_att_mlp_activation)
        self.title_state = AttendedState(title_att_mlp_layers, large_size2, title_att_dropout, title_att_mlp_activation)
        self.text_representation = SequentialModel([self.text_rnn, self.text_state])
        self.title_representation = SequentialModel([self.title_rnn, self.title_state])
        self.concatenation = ConcatenationLayer(self.text_representation, self.title_representation)
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size1 + large_size2,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, y, u):
        x = self.word_representation(x)
        y = self.word_representation(y)
        u = self.author_embedding_layer(u)
        u_x = u.unsqueeze(1).repeat(1, len(x[0]), 1)
        u_y = u.unsqueeze(1).repeat(1, len(y[0]), 1)
        x = torch.cat((x, u_x), 2)
        y = torch.cat((y, u_y), 2)
        x = self.text_representation(x)
        y = self.title_representation(y)
        output = self.concatenation(x, y)
        output = self.decision_layer(output)
        return output




