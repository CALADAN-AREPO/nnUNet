from nnunetv2.training.nnUNetTrainer.nnUNetTrainerHFF import nnUNetTrainerHFF


class nnUNetTrainerHFF_1200(nnUNetTrainerHFF):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=None):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1200
