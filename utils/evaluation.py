import json
from os.path import join
from collections import Counter
from data.processing import generate_char_tags, generate_word_tags

from data.processing import get_entity_emotion
from .decoding import flatten_lists, choose_core_entities, load_nerDict, load_complete_articles


class Metrics(object):
    """用于评价模型，计算每个标签的精确率，召回率，F1分数"""

    def __init__(self, golden_tags, predict_tags, remove_O=False):

        # [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.golden_tags = flatten_lists(golden_tags)
        self.predict_tags = flatten_lists(predict_tags)

        if remove_O:  # 将O标记移除，只关心实体标记
            self._remove_Otags()

        # 辅助计算的变量
        self.tagset = set(self.golden_tags)
        self.correct_tags_number = self.count_correct_tags()
        self.predict_tags_counter = Counter(self.predict_tags)
        self.golden_tags_counter = Counter(self.golden_tags)

        # 计算精确率
        self.precision_scores = self.cal_precision()

        # 计算召回率
        self.recall_scores = self.cal_recall()

        # 计算F1分数
        self.f1_scores = self.cal_f1()

    def cal_precision(self):

        precision_scores = {}
        for tag in self.tagset:
            precision_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                (self.predict_tags_counter[tag]+1e-10)

        return precision_scores

    def cal_recall(self):

        recall_scores = {}
        for tag in self.tagset:
            recall_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                (self.golden_tags_counter[tag]+1e-10)
        return recall_scores

    def cal_f1(self):
        f1_scores = {}
        for tag in self.tagset:
            p, r = self.precision_scores[tag], self.recall_scores[tag]
            f1_scores[tag] = 2*p*r / (p+r+1e-10)  # 加上一个特别小的数，防止分母为0
        return f1_scores

    def report_scores(self, outfile=None):
        """将结果用表格的形式打印出来，像这个样子：

                      precision    recall  f1-score   support
              B-LOC      0.775     0.757     0.766      1084
              I-LOC      0.601     0.631     0.616       325
             B-MISC      0.698     0.499     0.582       339
             I-MISC      0.644     0.567     0.603       557
              B-ORG      0.795     0.801     0.798      1400
              I-ORG      0.831     0.773     0.801      1104
              B-PER      0.812     0.876     0.843       735
              I-PER      0.873     0.931     0.901       634

          avg/total      0.779     0.764     0.770      6178
        """
        # 打印表头
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}\n'
        header = ['precision', 'recall', 'f1-score', 'support']
        reports = header_format.format('', *header)

        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}\n'
        # 打印每个标签的 精确率、召回率、f1分数
        for tag in self.tagset:
            reports += row_format.format(
                tag,
                self.precision_scores[tag],
                self.recall_scores[tag],
                self.f1_scores[tag],
                self.golden_tags_counter[tag]
            )

        # 计算并打印平均值
        avg_metrics = self._cal_weighted_average()
        reports += row_format.format(
            'avg/total',
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['f1_score'],
            len(self.golden_tags)
        )

        if outfile:
            outfile.write(reports)
        else:
            print(reports)

    def count_correct_tags(self):
        """计算每种标签预测正确的个数(对应精确率、召回率计算公式上的tp)，用于后面精确率以及召回率的计算"""
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            if gold_tag == predict_tag:
                if gold_tag not in correct_dict:
                    correct_dict[gold_tag] = 1
                else:
                    correct_dict[gold_tag] += 1

        return correct_dict

    def _cal_weighted_average(self):

        weighted_average = {}
        total = len(self.golden_tags)

        # 计算weighted precisions:
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for tag in self.tagset:
            size = self.golden_tags_counter[tag]
            weighted_average['precision'] += self.precision_scores[tag] * size
            weighted_average['recall'] += self.recall_scores[tag] * size
            weighted_average['f1_score'] += self.f1_scores[tag] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average

    def _remove_Otags(self):

        new_golden_tags = []
        new_predict_tags = []
        length = len(self.golden_tags)
        removed_count = 0

        for i in range(length):
            if not self.golden_tags[i].endswith('O'):
                new_golden_tags.append(self.golden_tags[i])
                new_predict_tags.append(self.predict_tags[i])
            else:
                removed_count += 1

        self.golden_tags = new_golden_tags
        self.predict_tags = new_predict_tags

        print("原总标记数为{}，移除了{}个O标记，占比{:.2f}%".format(
            length,
            removed_count,
            removed_count / length * 100
        ))

    def report_confusion_matrix(self, outfile=None):
        """计算混淆矩阵"""

        reports = "\nConfusion Matrix:\n"

        tag_list = list(self.tagset)
        # 初始化混淆矩阵 matrix[i][j]表示第i个tag被模型预测成第j个tag的次数
        tags_size = len(tag_list)
        matrix = []
        for i in range(tags_size):
            matrix.append([0] * tags_size)

        # 遍历tags列表
        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = tag_list.index(golden_tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:  # 有极少数标记没有出现在golden_tags，但出现在predict_tags，跳过这些标记
                continue

        # 输出矩阵
        row_format_ = '{:>7} ' * (tags_size+1) + '\n'
        reports += row_format_.format("", *tag_list)
        for i, row in enumerate(matrix):
            reports += row_format_.format(tag_list[i], *row)

        if outfile:
            outfile.write(reports)
        else:
            print(reports)


class EntityEmotionMetrics(object):
    """从比赛的评价指标评估模型"""

    def __init__(self, predict_tags, articles_truncated,
                 article_ids, token_method, split='dev'):

        self.predict_tags = predict_tags
        self.articles_truncated = articles_truncated
        self.article_ids = article_ids
        self.token_method = token_method
        self.split = split

        self.articles_completed, self.golden_tags = self._load_complete_articles()

        self._cal_scores()

    def report(self, outfile=None):
        outputs = "实体准确率: {:.3f}, 召回率: {:.3f}, F1: {:.4f}\n".format(
            self.entity_precision, self.entity_recall, self.entity_f1
        )
        outputs += "情感准确率: {:.3f}, 召回率: {:.3f}, F1: {:.4f}\n".format(
            self.motion_precision, self.motion_recall, self.motion_f1
        )
        outputs += "综合F1分数值：{:.4f}\n".format(
            self.final_f1
        )

        outputs += "（全部返回时）实体准确率: {:.3f}, 召回率: {:.3f}, F1: {:.4f}\n".format(
            self.all_entity_precision, self.all_entity_recall, self.all_entity_f1
        )
        outputs += "（全部返回时）情感准确率: {:.3f}, 召回率: {:.3f}, F1: {:.4f}\n".format(
            self.all_motion_precision, self.all_motion_recall, self.all_motion_f1
        )
        outputs += "（全部返回时）综合F1分数值：{:.4f}\n".format(
            self.all_final_f1
        )

        if outfile:
            outfile.write(outputs)
        else:
            print(outputs)

    def report_details(self, outfile=None):
        """更加详细的报告，
            主要是两个方面：
            1.准确率 召回率 f1, 混淆矩阵
            2.原文章,参与标记的部分,预测的标记,标准的标记, 预测的实体，标准实体，经过筛选的实体
        """

        # 1 标记 的 准确率 召回率 f1, 混淆矩阵
        outfile.write("未移除O标记时，模型得分以及混淆矩阵如下:\n")
        tag_metrics = Metrics(
            self.golden_tags, self.predict_tags, remove_O=False)
        tag_metrics.report_scores(outfile=outfile)
        tag_metrics.report_confusion_matrix(outfile=outfile)

        outfile.write("\n\n移除O标记后，模型得分以及混淆矩阵如下:\n")
        tag_metrics = Metrics(
            self.golden_tags, self.predict_tags, remove_O=True)
        tag_metrics.report_scores(outfile=outfile)
        tag_metrics.report_confusion_matrix(outfile=outfile)

        # 2
        self.report(outfile=outfile)
        outfile.write("\n\n下面提取的具体内容:\n")
        nerDict = load_nerDict()
        for gold, predict, art_truncated, id_ in zip(
            self.golden_tags,
            self.predict_tags,
            self.articles_truncated,
            self.article_ids
        ):
            art_completed, gold_entities, gold_motions = self.articles_completed[id_]
            all_pred_entities, all_pred_motions = get_entity_emotion(
                art_truncated, predict, all_return=True)
            pred_entities, pred_motions = choose_core_entities(
                all_pred_entities, all_pred_motions,
                art_completed,  nerDict
            )  # 经过筛选的 实体，看看比起全部返回，效果是否有提升

            outputs = "原文: {}\n\
            参与标记的部分: {}\n\
            标准标记: {}\n\
            预测的标记：{}\n\
            标准实体：{}\n\
            预测是全部实体: {}\n\
            经过筛选的实体: {}\n\
            ".format(
                art_completed,
                art_truncated,
                "\t".join(gold),
                "\t".join(predict),
                "\t".join(gold_entities+gold_motions),
                "\t".join(all_pred_entities+all_pred_motions),
                "\t".join(pred_entities+pred_motions)
            )
            outfile.write(outputs)
            outfile.write("="*50+'\n\n')  # 分割线

    def _cal_scores(self):
        """返回预测正确的实体个数，以及情感个数"""
        correct_entity = 0
        correct_emotion = 0
        all_correct_entity = 0
        all_correct_emotion = 0

        gold_sum = 0  # 总的个数
        pred_sum = 0  # 总的预测的个数(只取前3)
        all_pred_sum = 0  # 总的预测的实体个数

        nerDict = load_nerDict()
        for predict, art_truncated, id_ in zip(
            self.predict_tags,
            self.articles_truncated,
            self.article_ids
        ):

            art_completed, gold_entities, gold_motions = self.articles_completed[id_]
            all_pred_entities, all_pred_motions = get_entity_emotion(
                art_truncated, predict, all_return=True)
            pred_entities, pred_motions = choose_core_entities(
                all_pred_entities, all_pred_motions,
                art_completed,  nerDict
            )  # 经过筛选的 实体，看看比起全部返回，效果是否有提升

            gold_sum += len(gold_entities)
            pred_sum += len(pred_entities)
            all_pred_sum += len(all_pred_entities)

            # 计算正确的个数
            for pred_e, pred_m in zip(pred_entities, pred_motions):
                if pred_e in gold_entities:
                    correct_entity += 1
                    ind = gold_entities.index(pred_e)
                    if pred_m == gold_motions[ind]:
                        correct_emotion += 1

            for pred_e, pred_m in zip(all_pred_entities, all_pred_motions):
                if pred_e in gold_entities:
                    all_correct_entity += 1
                    ind = gold_entities.index(pred_e)
                    if pred_m == gold_motions[ind]:
                        all_correct_emotion += 1

        try:
            # pred_sum一开始可能为0，此时模型还未能从句子中抽取出实体
            self.entity_precision = correct_entity / pred_sum
            self.motion_precision = correct_emotion / pred_sum
            self.all_entity_precision = all_correct_entity / all_pred_sum
            self.all_motion_precision = all_correct_emotion / all_pred_sum
        except ZeroDivisionError:
            self.entity_precision = 1e-10  # 加上一个很小的数，防止它为0,因为下面计算f1的时候，需要它当分母
            self.motion_precision = 1e-10
            self.all_entity_precision = 1e-10
            self.all_motion_precision = 1e-10

        self.entity_recall = correct_entity / gold_sum + 1e-10
        self.motion_recall = correct_emotion / gold_sum + 1e-10
        self.all_entity_recall = all_correct_entity / gold_sum + 1e-10
        self.all_motion_recall = all_correct_emotion / gold_sum + 1e-10

        self.entity_f1 = 2 / (1./self.entity_precision + 1./self.entity_recall)
        self.motion_f1 = 2 / (1./self.motion_precision + 1./self.motion_recall)
        self.all_entity_f1 = 2 / \
            (1./self.all_entity_precision + 1./self.all_entity_recall)
        self.all_motion_f1 = 2 / \
            (1./self.all_motion_precision + 1./self.all_motion_recall)

        self.final_f1 = 0.5 * (self.entity_f1 + self.motion_f1)
        self.all_final_f1 = 0.5 * (self.all_entity_f1 + self.all_motion_f1)

    def _load_complete_articles(self, data_dir="SohuData/original_data"):
        """加载完整的文章，用于筛选核心实体"""
        assert self.split in ['train', 'dev']

        data_file = join(data_dir, "{}_{}.txt".format(
            self.split, self.token_method))
        articles = {}
        golden_tags = [[]] * len(self.article_ids)
        with open(data_file) as f:
            for line in f:
                json_data = json.loads(line)
                if self.token_method == "char":
                    article, tags = generate_char_tags(json_data)
                else:
                    article, tags = generate_word_tags(json_data)
                    article = "".join(article)

                entities = []
                emotions = []
                for item in json_data['coreEntityEmotions']:
                    entities.append(item['entity'])
                    emotions.append(item['emotion'])
                articles[json_data['newsId']] = [article, entities, emotions]

                ind = self.article_ids.index(json_data['newsId'])
                golden_tags[ind] = tags[:len(self.predict_tags[ind])]

        return articles, golden_tags
