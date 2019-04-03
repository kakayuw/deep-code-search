import numpy as np
import matplotlib.pyplot as plt

deepcs_top1_acc = 0.46
deepcs_mrr = 0.60
deepcs_top10_acc = 0.86
deepcs_top10_map = 0.49



def draw_mrr_acc_valid( inputfile, outputfile, author):
    epoch, acc, mrr = [], [], []
    with open(inputfile) as f:
        for line in f:
            epoch.append(int(line.split(' ')[0]))
            acc.append(round(float(line.split(' ')[1]), 3))
            mrr.append(round(float(line.split(' ')[2]), 3))

    t = np.arange(0, 69, 1)

    epoch = np.array(epoch)
    acc = np.array(acc)
    mrr = np.array(mrr)
    print (epoch)
    print (acc)
    print (mrr)
    plt.plot(epoch, acc, color="blue")
    plt.plot(epoch, mrr, color="red")
    plt.plot(epoch, np.full((len(acc),), deepcs_top1_acc), color='darkblue')
    plt.plot(epoch, np.full((len(mrr),), deepcs_mrr), color='darkred')
    label = ['acc', 'mrr', 'deepcs top1 acc', 'deepcs mrr']
    plt.legend(label, loc='lower right')
    plt.title(author + ": tok1 metrics on validating dataset 2000")
    plt.xlabel("epoch")
    plt.ylabel("rate")
    plt.savefig(outputfile)
    plt.show()


def draw_eval( inputfile, outputfile, author):
    epoch, acc, mrr, map, ndcg = [], [], [], [], []
    with open(inputfile) as f:
        for line in f:
            epoch.append(int(line.split(' ')[0]))
            acc.append(round(float(line.split(' ')[1]), 3))
            mrr.append(round(float(line.split(' ')[2]), 3))
            map.append(round(float(line.split(' ')[3]), 3))
            ndcg.append(round(float(line.split(' ')[4]), 3))

    epoch = np.array(epoch)
    acc = np.array(acc)
    mrr = np.array(mrr)
    map = np.array(map)
    ndcg = np.array(ndcg)

    plt.plot(epoch, acc, color="blue")
    plt.plot(epoch, mrr, color="red")
    plt.plot(epoch, map, color="green")
    plt.plot(epoch, ndcg, color="gold")
    plt.plot(epoch, np.full((len(acc),), deepcs_top10_acc), color='darkblue')
    plt.plot(epoch, np.full((len(mrr),), deepcs_mrr), color='darkred')
    plt.plot(epoch, np.full((len(map),), deepcs_top10_map), color='darkgreen')
    # plt.plot(epoch, np.full((len(ndcg),), ), color='darkgoldenrod')
    label = ['acc', 'mrr', 'map', 'ndcg', 'deepcs top1 acc', 'deepcs mrr', 'deepcs map']
    plt.legend(label, loc='lower right')
    plt.title(author + ": tok 10 metrics on evaluating dataset 20000")
    plt.xlabel("epoch")
    plt.ylabel("rate")
    plt.savefig(outputfile)
    plt.show()

if __name__ == '__main__':

    yh_allepoch = "./log/yh_valid.log"
    yh_allepoch_pic = './log/yh_valid_trend.jpg'
    heyq_allepoch = "./log/heyq_valid.log"
    heyq_allepoch_pic = './log/heyq_valid_trend.jpg'
    yh_eval = "./log/yh_eval12.log"
    yh_eval_pic = './log/yh_eval12.jpg'
    yh = "yh"
    hyq = "hyq"

    # draw_mrr_acc_valid(yh_allepoch, yh_allepoch_pic, yh)
    # draw_mrr_acc_valid(heyq_allepoch, heyq_allepoch_pic, hyq)

    draw_eval(yh_eval, yh_eval_pic, yh)
