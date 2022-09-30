import sys
sys.path.append('../')
from network_make_submit import NetworkInference
from options import Options

def main():
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    opt.root_dir = '/media/alexey/HDDDataDIsk/datasets/RetinalVesselSegmentation/CompetitionDataset/test/'

    inferencer = NetworkInference(opt)
    inferencer.set_GPU_device()
    inferencer.set_network()
    inferencer.set_dataloader()
    inferencer.set_save_dir()
    inferencer.run()


if __name__ == "__main__":
    main()
