import os.path as osp
import mmengine
from mmengine.runner import Runner
from mmengine import Config
from mmengine.runner import set_random_seed



if __name__ == "__main__":
# find 
    import pyrootutils
    from omegaconf import DictConfig
    import hydra
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs")
    output_path = path / "outputs"
    print("paths", path, config_path, output_path)







    @hydra.main(version_base="1.3", config_path=config_path, config_name="train.yaml")
    def main(cfg: DictConfig):
        config = Config.fromfile(cfg.config_file)
        config.data_root = cfg.data.data_root
        config.data_root_val = cfg.data.data_root_val
        config.ann_file_train = cfg.data.ann_file_train
        config.ann_file_val = cfg.data.ann_file_val

        # config.test_dataloader.dataset.ann_file = cfg.data.ann_file_test
        # config.test_dataloader.dataset.data_prefix.video = cfg.data.data_prefix_video_test

        config.train_dataloader.dataset.ann_file = cfg.data.ann_file_train
        config.train_dataloader.dataset.data_prefix.video = cfg.data.data_root

        config.val_dataloader.dataset.ann_file = cfg.data.ann_file_val
        config.val_dataloader.dataset.data_prefix.video  = cfg.data.data_root_val

        config.model.cls_head.num_classes = cfg.model.num_classes
    # We can use the pre-trained TSN model
        config.load_from = cfg.model.checkpoint_file


        config.work_dir = cfg.work_dir
        config.train_dataloader.batch_size = cfg.train_batch_size
        config.val_dataloader.batch_size = cfg.val_batch_size
        config.train_cfg.max_epochs = cfg.max_epochs
        config.train_dataloader.num_workers = 2
        config.val_dataloader.num_workers = 2
        


        mmengine.mkdir_or_exist(osp.abspath(config.work_dir))
        print(config.train_dataloader.batch_size)

        # build the runner from config
        runner = Runner.from_cfg(config)
        # print(runner)

        # start training
        runner.train()


        # set_data(config,cfg)
        # print(config)
        # test_net(cfg)
        # test_module(cfg)

    # Create work_dir
    
    main()