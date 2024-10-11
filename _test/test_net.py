from mynn.cluster import clustering
from mynn.siam_train import train_siamese
from mynn.spec_ds import get_mv_dataloader
from mynn.spec_train import train_spectral
from utils.SL import load_config, load_object


def train_model(ds_name):
    cfg_path = '../res/config/' + ds_name + '.yaml'
    ds_from_path = '../res/' + ds_name + '/'
    save_path = '../res/' + ds_name + '/'
    cfg = load_config(cfg_path)
    train_siamese(cfg, ds_from_path, save_path)
    train_spectral(cfg, ds_from_path, save_path)


def test_cluster(ds_name):
    test_path = '../res/' + ds_name + '/test_mv.b'
    model_path = '../res/' + ds_name + '/spectral_net_mv.model'
    dloader = get_mv_dataloader(test_path, batch_size=1000)
    model = load_object(model_path)
    for b_idx, (mv_data_batch, label_batch) in enumerate(dloader):
        mv_out_batch = model(mv_data_batch)
        mv_out_batch = [data.cpu().detach().numpy() for data in mv_out_batch]
        label_batch = label_batch.cpu().detach().numpy()
        pred_batch = clustering(mv_out_batch, label_batch)

# train_model('NoisyMNIST')
# test_cluster('NoisyMNIST')
