import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset


class RUGD4Dataset(BaseDataset):
    def __init__(
        self,
        dataroot,
        annpath,
        trans_func=None,
        mode="train",
    ):
        super().__init__(
            dataroot,
            annpath,
            trans_func,
            mode,
        )

        self.lb_ignore = 255

        # 替换成 compute_rugd_stats.py 的真实输出
        self.to_tensor = T.ToTensor(
            mean=(0.4032165247740185, 0.40301203945244995, 0.40141448779249095),
            std=(0.2762801330026067, 0.27541393419064897, 0.28151901441168853),
        )