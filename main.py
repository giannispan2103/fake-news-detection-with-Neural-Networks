from models import TextRNN, ProjectedAttentionTextTitleAuthorRNN
from preprocess import generate_data
from utils import train
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss


def run_text():
    data = generate_data(split_point=15000, emb_size=50, maxlen=100)
    emb_matrix = data['emb_matrix']
    train_batches = data['train_batches']
    test_batches = data['test_batches']
    model = TextRNN(emb_matrix, trainable_embeddings=False)
    optimizer = Adam(model.params, 0.001)
    criterion = BCEWithLogitsLoss()
    train(model, train_batches, test_batches, optimizer, criterion, 50, 5)


def run_text_title_author():
    data = generate_data(split_point=15000, emb_size=50, maxlen=25)
    a2i = data['a2i']
    emb_matrix = data['emb_matrix']
    train_batches = data['train_batches']
    test_batches = data['test_batches']
    model = ProjectedAttentionTextTitleAuthorRNN(emb_matrix, author_embeddings_input_size=len(a2i), embeddings_dropout=0.5, top_mlp_dropout=0.5, text_stacked_layers=1, text_cell_hidden_size=128,
                                                 title_cell_hidden_size=32, top_mlp_outer_activation=None, top_mlp_layers=2)
    optimizer = Adam(model.params, 0.001)
    criterion = BCEWithLogitsLoss()
    train(model, train_batches, test_batches, optimizer, criterion, 50, 5)


if __name__ == "__main__":
    run_text_title_author()
