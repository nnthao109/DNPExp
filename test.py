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



    @hydra.main(version_base="1.3", config_path=config_path, config_name="test.yaml")
    def main(cfg: DictConfig):
        config = Config.fromfile(cfg.config_file)
        
        config.test_dataloader.dataset.ann_file = cfg.ann_file_test
        config.test_dataloader.dataset.data_prefix.video = cfg.data_prefix_video_test
        config.work_dir = './tutorial_exps'
        # mmengine.mkdir_or_exist(osp.abspath(config.work_dir))
        # print(config.train_dataloader.batch_size)

        # build the runner from config
        runner = Runner.from_cfg(config)
        # print(runner)

        # start training
        runner.test()


    # Create work_dir
    
    main()