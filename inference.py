import os.path as osp
import mmengine
from mmaction.apis import inference_recognizer, init_recognizer
from mmengine import Config
from operator import itemgetter
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



    @hydra.main(version_base="1.3", config_path=config_path, config_name="inference.yaml")
    def main(cfg: DictConfig):
        config = Config.fromfile(cfg.config_file)
        
        checkpoint = cfg.checkpoint_file
        device =cfg.device

        video = cfg.video
        label = cfg.label


        model = init_recognizer(config, checkpoint, device)

        results = inference_recognizer(model, video)

        pred_scores = results.pred_scores.item.tolist()
        score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
        score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
        top5_label = score_sorted[:]

        labels = open(label).readlines()
        labels = [x.strip() for x in labels]
        # print(top5_label)
        # print(labels)
        results = [(labels[k[0]], k[1]) for k in top5_label]

        print('The top-5 labels with corresponding scores are:')
        for result in results:
            print(f'{result[0]}: ', result[1])


    # Create work_dir
    
    main()

