import datetime
import pandas as pd
import torch
import os
import sklearn.metrics as metrics
from dataloader import label_to_idx


class IOStream():
    """
    Logging to screen and file
    """
    def __init__(self, args):
        self.path = args.tb_log_dir
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.f = open(self.path + '/run_{}_{}_{}'.format(args.source, 
                                                         args.target, 
                                                         1) + 'ver13.log', 'w')
        self.args = args

    def cprint(self, text):
        datetime_string = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
        to_print = "%s: %s" % (datetime_string, text)
        print(to_print)
        self.f.write(to_print + "\n")
        self.f.flush()

    def close(self):
        self.f.close()

    def save_model(self, model):
        path = self.path + '/{}_{}_{}'.format(self.args.source, 
                                              self.args.target, 
                                              1) +'ver13.pt'
        torch.save(model.state_dict(), path)
        return

    def save_conf_mat(self, conf_matrix, fname, domain_set):
        df = pd.DataFrame(conf_matrix, columns=list(label_to_idx.keys()), index=list(label_to_idx.keys()))
        fname = domain_set + "_" + fname
        df.to_csv(self.path + "/" + fname)

    def print_progress(self, domain_set, partition, epoch, print_losses, true=None, pred=None):
        outstr = "%s - %s %d" % (partition, domain_set, epoch)
        acc = 0
        if true is not None and pred is not None:
            acc = metrics.accuracy_score(true, pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(true, pred)
            outstr += ", acc: %.4f, avg acc: %.4f" % (acc, avg_per_class_acc)

        for loss, loss_val in print_losses.items():
            outstr += ", %s loss: %.4f" % (loss, loss_val)
        self.cprint(outstr)
        return acc
