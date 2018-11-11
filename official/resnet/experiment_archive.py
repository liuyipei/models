"""
python temp.py > slurm/experiments_0810.txt
python make_experiment.py > slurm/experiments_0816_origlike.txt
python make_experiment.py > slurm/experiments_0817_best_mm_gc.txt
python make_experiment.py > slurm/experiments_0817_resnet68.txt
python make_experiment.py > slurm/experiments_0817_b1.txt
python make_experiment.py > slurm/experiments_0817_imagenet.txt
python make_experiment.py > slurm/experiments_0818_b1_slow.txt
python make_experiment.py > slurm/experiments_0818_b1_mmm.txt ## been redone with 64 bit accum -- see 0919_b1_mmmls
python make_experiment.py > slurm/experiments_0819_b8.txt
python make_experiment.py > slurm/experiments_0819_b1_lscale.txt
python make_experiment.py > slurm/experiments_0819_b1_mmmls.txt ## results written to 0818_b1_mmm . still has increasing test-entropy (though also improving test accuracy)
python make_experiment.py > slurm/experiments_0820_b8_rvs.txt
python make_experiment.py > slurm/experiments_0820_imagenet50.txt
python make_experiment.py > slurm/experiments_0820_b1_lscale.txt
python make_experiment.py > slurm/experiments_0820_imagenet50redo.txt
python make_experiment.py > slurm/experiments_0820_imagenet50redo2.txt
python make_experiment.py > slurm/experiments_0821_b4.txt
python make_experiment.py > slurm/experiments_0821_b8_rvs_slow.txt   # GTX -- partially running. somewhat interesting at 512 denom speed 
python make_experiment.py > slurm/experiments_0821_b8_ctrl_slow.txt  # GTX -- cancelled before start
python make_experiment.py > slurm/experiments_0822_b8_bid.txt        # GTX -- cancelled before start
python make_experiment.py > slurm/experiments_0822_b8_bid2em5.txt  # K80 -- rvs not use. 128 denom might be too fast.
python make_experiment.py > slurm/experiments_0822_b64_imagenet50.txt   ## too fast for me at original denom
python make_experiment.py > slurm/experiments_0822_b64_imagenet50slowbid2em5.txt # P100 # set denom to 512
python make_experiment.py > slurm/experiments_0822_b64_imagenet50slowctrl.txt # P100 # set denom to 512
python make_experiment.py > slurm/experiments_0822_b64_imagenet50slowbid2em6.txt # P100 # set denom to 512
python make_experiment.py > slurm/experiments_0823_b8_ctrl.txt  # GTX -- PD (3 jobs) [256 denom]
python make_experiment.py > slurm/experiments_0823_b8_bid2em456.txt  # GTX -- PD (15 jobs) [256 denom]
python make_experiment.py > slurm/experiments_0823_b8_bfnd2em456.txt  # GTX -- PD (12 jobs) [128 denom]
python make_experiment.py > slurm/experiments_0823_b8_ctrl128.txt  # GTX -- PD (3 jobs) [256 denom]
python make_experiment.py > slurm/experiments_0824_b8_inf.txt  # 10 jobs on K80. 128 denom.
python make_experiment.py > slurm/experiments_0825_b8_loo.txt  # 10 jobs on K80. 128 denom.
python make_experiment.py > slurm/experiments_0825_b8_loo_d512.txt  # 10 jobs on K80. 128 denom.
python make_experiment.py > slurm/experiments_0825_b8_inf.txt  # 4 jobs on GTX. 128 denom.
python make_experiment.py > slurm/experiments_0826_b8_loof.txt  # 4 on gtx, 4 moew pending, out of 8

python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0817_best_mm_gc.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0817_resnet68.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0817_b1.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0817_imagenet.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0818_b1_slow.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0818_b1_mmm.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0819_b8.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0819_b1_lscale.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0820_b8_rvs.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0820_imagenet50.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0820_b1_lscale.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0820_imagenet50redo.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0820_imagenet50redo2.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0821_b4.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0821_b8_rvs_slow.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0821_b8_ctrl_slow.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0822_b8_bid.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0822_b8_bid2em5.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0822_b64_imagenet50.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0822_b64_imagenet50slowctrl.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0822_b64_imagenet50slowbid2em6.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0822_b64_imagenet50slowbid2em5.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0823_b8_ctrl.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0823_b8_bid2em456.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0823_b8_bfnd2em456.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0823_b8_ctrl128.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0824_b8_inf.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0825_b8_loo.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0825_b8_inf.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0826_b8_loof.txt --src=~/gpuenv/activate

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
a= tf.Variable([], dtype=tf.float32)

python imagenet_main.py --data_dir=/home/yiliu/calico/data/imagenet -md=basic_resnet # to resume this
python imagenet_main.py --data_dir=imagenet_data -md=0817_imagenet

cifar10 now with eps=1e-3
batchfreesigf mmfwd=mmgrad=[0., .5, .9]
batchfreesigf mmfwd=.997 mmgrad=[0., .5, .9]

imagenet experiments:
    identity
    baselinebn
    batchfreesigf grad_momentum=0.
    batchfreesigf grad_momentum=.5
    batchfreesigf grad_momentum=.9
    b=1, 2, 128
"""



bn_shortthand_dict = {
    'tf_layers_regular': 'baselinebn', 
    'tf_layers_renorm': 'renorm', 
    'batch_free_normalization_sigfunc': 'bfsigf',
    'batch_free_normalization_sigconst': 'bfsigc',
    'batch_free_normalization_sigfunc_compare_running_stats': 'bfsigfcmp',
    'identity': 'identity'}


def make_cifar10_experiments_0810():

    series = '0810'
    dd="./cifar10_data"
    stderr='stderr'
    md_list = []
    for batch_size in [128, 2, 1]:
        # bnmethods_to_use = ['tf_layers_regular', 'tf_layers_renorm', 'batch_free_normalization_compare_running_stats']
        bnmethods_to_use = ['tf_layers_regular', 'tf_layers_renorm', 'batch_free_normalization', 'batch_free_normalization_sigfunc', 'identity']
        for bnmethod in bnmethods_to_use:
            shorthand = bn_shortthand_dict[bnmethod]
            shorthand_b = "%s_b%0.3d" % (shorthand, batch_size)
            md = "./%s/cifar10_%s" % (series, shorthand_b)
            bs = "%d" % batch_size
            trailing_pipe = ">%s/%s_cifar10_%s.out 2>%s/%s_cifar10_%s.err" % (stderr, series, shorthand_b, stderr, series, shorthand_b)
            cmd = "stdbuf -oL python3 cifar10_main.py -dd=%s -md=%s -bs=%s -bnmethod=%s %s" % \
                (dd, md, bs, bnmethod, trailing_pipe)
            print(cmd)
            md_list.append(md)

def make_cifar10_experiments_0814():

    series = '0814'
    dd="./cifar10_data"
    stderr='stderr'
    md_list = []
    for batch_size in [128, 2, 1]:
        bnmethods_to_use = ['batch_free_normalization_sigfunc']
        for bnmethod in bnmethods_to_use:
            for mmfwd, mmgrad in [(.5, .5), 
                                  (.9, .9), 
                                  (.9, .5),
                                  (.997, .997),
                                  (.997, .5),
                                  (.997, 0.)]:
                mmfwd_str = "%0.3f" % mmfwd
                mmgrad_str = "%0.3f" % mmgrad
                shorthand = bn_shortthand_dict[bnmethod]
                shorthand_b = "%s_b%0.3d_mmfwd_%s_mmgrad_%s" % (shorthand, batch_size, mmfwd_str, mmgrad_str)
                md = "./%s/cifar10_%s" % (series, shorthand_b)
                bs = "%d" % batch_size
                trailing_pipe = ">%s/%s_cifar10_%s.out 2>%s/%s_cifar10_%s.err" % (stderr, series, shorthand_b, stderr, series, shorthand_b)
                cmd = "stdbuf -oL python3 cifar10_main.py -dd=%s -md=%s -bs=%s -bnmethod=%s -mmfwd=%s -mmgrad=%s %s" % \
                    (dd, md, bs, bnmethod, mmfwd_str, mmgrad_str, trailing_pipe)
                print(cmd)
                md_list.append(md)

def make_cifar10_experiments_0815_helper(**kwargs):
    make_experiments_0815_helper(dataset='cifar10', **kwargs)

def make_experiments_0815_helper(
        batch_size_list = (2,),
        bnmethods_list = ('batch_free_normalization_sigfunc',),
        bfn_mm_list = ((.5, .5), 
                       (.9, .9), 
                       (.9, .5),
                       (.997, .997),
                       (.997, .5),
                       (.997, 0.)),
        opt_mm_list = (0., ),
        batch_denom_list=(64., ),
        bfn_grad_clip_list=(None,),
        series='0815',
        resnet_size=None,
        dataset='cifar10',
        loss_scale=None,
        single_bfn_mm_short=None,
        rvs_list=(None,),
        ):
    dd="./%s_data" % dataset
    stderr='stderr'

    md_list = []
    import itertools
    opts_gen = itertools.product(
        batch_size_list, bnmethods_list, bfn_mm_list, opt_mm_list, batch_denom_list, bfn_grad_clip_list, rvs_list)
    for batch_size,      bnmethod,   (mmfwd, mmgrad), opt_mm,      batch_denom,      grad_clip,          rvs in opts_gen:
        mmfwd_str = "%0.6f" % mmfwd
        mmgrad_str = "%0.6f" % mmgrad
        opt_mm_str = "%0.6f" % opt_mm
        bs = "%d" % batch_size
        shorthand = bn_shortthand_dict[bnmethod]
        if grad_clip is not None:
            grad_clip_str = "%0.3f" % grad_clip
            grad_clip_short = "_gc_%s" % grad_clip_str
        else:
            grad_clip_short = ""

        if single_bfn_mm_short is not None:
            bfn_mm_short = single_bfn_mm_short
            assert len(bfn_mm_list) == 1
        else:
            bfn_mm_short = "mmf_%s_mmg_%s" % (mmfwd_str, mmgrad_str)
        if rvs is not None:
            rvs_t, rvs_v = rvs
            assert rvs_v is None
            rvs_short = "_%s" % rvs_t
        else:
            rvs_str, rvs_short = "", ""

        shorthand_b = "%s_b%0.3d_%s_bdnm_%d_omm_%s%s%s" % \
            (shorthand, batch_size, bfn_mm_short, batch_denom, opt_mm_str, grad_clip_short, rvs_short)
        md = "./%s/%s_%s" % (series, dataset, shorthand_b)
        cmd_tokens = [
            'stdbuf -oL python3 %s_main.py' % dataset,
            '-dd=%s' % dd,
            '-md=%s' % md,
            '-bs=%s' % bs,
            '-bnmethod=%s' % bnmethod,
            '-mmfwd=%s' % mmfwd_str,
            '-mmgrad=%s' % mmgrad_str,
            '-opt_mm=%s' % opt_mm,
            '-batch_denom=%s' % batch_denom, ]
        if grad_clip is not None:
            cmd_tokens.append('-bfn_grad_clip=%s' % grad_clip_str)
        if resnet_size is not None:
            cmd_tokens.append('-resnet_size=%d' % resnet_size)
        if loss_scale is not None:
            cmd_tokens.append('-loss_scale=%d' % loss_scale)
        if rvs is not None:
            cmd_tokens.append('-rvst=%s' % rvs_t)
        
        trailing_pipe = ">%s/%s_%s_%s.out 2>%s/%s_%s_%s.err" % (stderr, series, dataset, shorthand_b, \
                                                                stderr, series, dataset, shorthand_b)
        cmd_tokens.append(trailing_pipe)
        cmd = ' '.join(cmd_tokens)
        #cmd = "stdbuf -oL python3 cifar10_main.py -dd=%s -md=%s -bs=%s -bnmethod=%s -mmfwd=%s -mmgrad=%s -batch_denom=%d -opt_mm=%s %s" % \
        #    (dd, md, bs, bnmethod, mmfwd_str, mmgrad_str, batch_denom, opt_mm_str, trailing_pipe)
        print(cmd)
        md_list.append(md)

def make_cifar10_experiments_0815():
    return make_cifar10_experiments_0815_helper(batch_size_list = (128, 32, 2))

def make_cifar10_experiments_0815_b1():
    return make_cifar10_experiments_0815_helper(
        batch_size_list=(1, ),
        opt_mm_list=(0., .5, .9),
        bfn_mm_list = ((.5, .5), 
                       (.9, .9), 
                       (.9, .5),
                       (.997, .997),
                       (.997, .5)),
        series='0815_b1'
        )
def make_cifar10_experiments_0816_b1():
    return make_cifar10_experiments_0815_helper(
        batch_size_list=(1, ),
        opt_mm_list=(0., .9),
        bfn_mm_list = (
                       (.9, .9),
                       (.997, .997),
                      ),
        series='0816_b1',
        bfn_grad_clip_list=(1., 3., 10., 30.)
        )
def make_cifar10_experiments_0816_origlike():
    return make_cifar10_experiments_0815_helper(
        batch_size_list=(128, ),
        opt_mm_list=(0., .9),
        bfn_mm_list = (
                       (.5, .5),
                       (.9, .9),
                       (.997, .997),
                      ),
        series='0816_origlike',
        batch_denom_list=(128., ),
        bfn_grad_clip_list=(3., 10., 30., 100., 300., 1000.)
        )

def make_cifar10_experiments_0817_b1():
    # renamed summary variable to train_cross entropy to avoid name collision
    series = '0817_b1'
    make_cifar10_experiments_0815_helper(
        batch_size_list=(1, ),
        opt_mm_list=(.9, ),
        bfn_mm_list = (
                       (.5, .5),
                       (.9, .9),
                       (.997, .997),
                      ),
        series=series,
        batch_denom_list=(128., 256.,),
        bfn_grad_clip_list=(1., 10.)
        )
    make_cifar10_experiments_0815_helper(
        batch_size_list=(1, ),
        opt_mm_list=(.9, ),
        bnmethods_list = ('identity', 'batch_free_normalization_sigconst', 'tf_layers_regular', 'tf_layers_renorm'),
        bfn_mm_list = ((.997, .997),),
        series=series,
        batch_denom_list=(128., ),
        )

def make_cifar10_experiments_0817_helper(series = '0817', resnet_size=None):
    # renamed summary variable to train_cross entropy to avoid name collision
    make_cifar10_experiments_0815_helper(
        batch_size_list=(128, ),
        bnmethods_list = ('batch_free_normalization_sigfunc',),
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (.5, .5),
                       (.9, .9),
                       (.997, .5),
                       (.997, .997),
                       (.997, .5),
                      ),
        batch_denom_list=(128., ),
        bfn_grad_clip_list=(100., ),
        series=series,
        resnet_size=resnet_size,
        )
    make_cifar10_experiments_0815_helper(
        batch_size_list=(128, ),
        opt_mm_list=(.9,),
        bnmethods_list = (['identity','tf_layers_renorm','tf_layers_regular']),
        bfn_mm_list = ((.997, .997),),
        batch_denom_list=(128., ),
        bfn_grad_clip_list=(100., ),
        series=series,
        resnet_size=resnet_size,
        )

def make_cifar10_experiments_0817_best_mm_gc():
    make_cifar10_experiments_0817_helper(series = '0817_best_mm_gc')

def make_cifar10_experiments_0817_resnet68(resnet_size=None):
    make_cifar10_experiments_0817_helper(series = '0817_resnet68', resnet_size=68)

def make_imagenet_experiments_0817_imagenet101():
    make_experiments_0815_helper(dataset='imagenet',
        batch_size_list=(32, ),
        bnmethods_list = ('batch_free_normalization_sigfunc', 'tf_layers_renorm', 'tf_layers_regular', 'identity'),
        bfn_mm_list=((.997, .997), ),
        opt_mm_list = (.9, ),
        batch_denom_list=(256., ),
        bfn_grad_clip_list=(100.,),
        series='imagenet101',
        resnet_size=101,
        )

def make_cifar10_experiments_0818_b1_slow():
    # renamed summary variable to train_cross entropy to avoid name collision
    # inconclusive results of 16 experiments on k80
    series = '0818_b1_slow'
    make_cifar10_experiments_0815_helper(
        batch_size_list=(1, ),
        bnmethods_list = ('batch_free_normalization_sigfunc',),
        opt_mm_list=(.99, .999),
        bfn_mm_list = (
                       (.5, .5),
                       (.9, .9),
                       (.997, .997),
                      ),
        series=series,
        batch_denom_list=(128., ),
        bfn_grad_clip_list=(100., )
        )
    make_cifar10_experiments_0815_helper(
        batch_size_list=(1, ),
        bnmethods_list = ('batch_free_normalization_sigfunc',),
        opt_mm_list=(.9, ),
        bfn_mm_list = (
                       (.5, .5),
                       (.9, .9),
                       (.997, .997),
                      ),
        series=series,
        batch_denom_list=(512., 1024.),
        bfn_grad_clip_list=(100., )
        )
    make_cifar10_experiments_0815_helper(
        batch_size_list=(1, ),
        opt_mm_list=(.99, ),
        bnmethods_list = ('identity', 'batch_free_normalization_sigconst', 'tf_layers_regular', 'tf_layers_renorm'),
        bfn_mm_list = ((.997, .997),),
        series=series,
        batch_denom_list=(128., ),
        )

def make_cifar10_experiments_0818_b1_mmm():
    # matching momentum. inclusive results on 1080ti
    series = '0818_b1_mmm'
    for mm in [.999, .9997, .9999, .99997,]:
        make_cifar10_experiments_0815_helper(
            batch_size_list=(1, ),
            bnmethods_list = ('batch_free_normalization_sigfunc',),
            opt_mm_list=(mm, ),
            bfn_mm_list = (
                           (mm, mm),
                          ),
            series=series,
            batch_denom_list=(128., ),
            bfn_grad_clip_list=(100., None,)
            )

def make_cifar10_experiments_0819_b8():
    # summarize only running variances now. 6+3 = 9 runs
    # executing on gtx1080ti
    series = '0819_b8'
    make_cifar10_experiments_0815_helper(
        batch_size_list=(8, ),
        bnmethods_list = ('batch_free_normalization_sigfunc',),
        opt_mm_list=(.9, .99),
        bfn_mm_list = (
                       (.5, .5),
                       (.9, .9),
                       (.997, .997),
                      ),
        series=series,
        batch_denom_list=(128.,),
        )
    make_cifar10_experiments_0815_helper(
        batch_size_list=(8, ),
        opt_mm_list=(.9, ),
        bnmethods_list = ('identity', 'tf_layers_regular', 'tf_layers_renorm'),
        bfn_mm_list = ((.997, .997),),
        series=series,
        batch_denom_list=(128., ),
        )

def make_cifar10_experiments_0819_b1_lscale():
    # using loss_scale to guard against underflow. 2*3*2+4=16 runs
    # redo some basic earlier experiments. bfn grad clip needs ot be compensated along with loss scale
    # canceled -- then rerun, but with bfn now using float64 accumulators
    series = '0819_b1_lscale'
    make_cifar10_experiments_0815_helper(
        batch_size_list=(1, ),
        opt_mm_list=(.9, .99,),
        bfn_mm_list = (
                       (.5, .5),
                       (.9, .9),
                       (.997, .997),
                      ),
        series=series,
        batch_denom_list=(128.,),
        bfn_grad_clip_list=(8192, 32768),
        loss_scale=256,
        )
    make_cifar10_experiments_0815_helper(
        batch_size_list=(1, ),
        opt_mm_list=(.9, ),
        bnmethods_list = ('identity', 'batch_free_normalization_sigconst', 'tf_layers_regular', 'tf_layers_renorm'),
        bfn_mm_list = ((.997, .997),),
        series=series,
        batch_denom_list=(128., ),
        loss_scale=256,
        )

def make_cifar10_experiments_0819_b1_mmmls():
    # matching momentum. inclusive results on 1080ti, on float 64 bfn.
    # check quarter learning rate. do not make optmizer momentum higher  since i think it uses float 32 accumulator
    series = '0818_b1_mmm'
    for mm in [.997, .999, .9997, .9999,]:
        make_cifar10_experiments_0815_helper(
            batch_size_list=(1, ),
            bnmethods_list = ('batch_free_normalization_sigfunc',),
            opt_mm_list=(.997, ),
            bfn_mm_list = (
                           (mm, mm),
                          ),
            series=series,
            batch_denom_list=(128., 512.,),
            bfn_grad_clip_list=(32768, ),
            loss_scale=256,
            )

#### 0820-0826



bn_shortthand_dict = {
    'tf_layers_regular': 'baselinebn', 
    'tf_layers_renorm': 'renorm', 
    'identity': 'identity',
    'batch_free_normalization_sigfunc': 'bfsigf',
    'batch_free_normalization_sigconst': 'bfsigc',
    'batch_free_normalization_sigfunc_compare_running_stats': 'bfsigfcmp',
    'batch_free_direct': 'bfnd',
    'bfn_like_regular': 'bfnlikereg',
    'bfn_like_regular_ivtc_batch': "regbatch", 
    'bfn_like_loo_ivtc_batch': "loobatch",
    'bfn_like_regular_ivtc_running': "regrunning",
    'bfn_like_loo_ivtc_running': "loorunning",
    'bfn_like_loo_ivtc_loo': "looloo",
    }


def make_cifar10_experiments_0820_helper(**kwargs):
    make_experiments_0820_helper(dataset='cifar10', **kwargs)

def make_experiments_0820_helper(
        batch_size_list = (1,),
        bnmethods_list = ('batch_free_normalization_sigfunc',),
        bfn_mm_list = (
                       (.997, .997),
                       ),
        opt_mm_list = (.9, ),
        batch_denom_list=(128., ),
        bfn_grad_clip_list=(None,),
        series='0820',
        resnet_size=None,
        dataset='cifar10',
        loss_scale=None,
        single_bfn_mm_short=None,
        rvs_list=(None,),
        bfn_input_decay=None,
        ):
    dd="./%s_data" % dataset
    stderr='stderr'

    md_list = []
    import itertools
    opts_gen = itertools.product(
        batch_size_list, bnmethods_list, bfn_mm_list, opt_mm_list, batch_denom_list, bfn_grad_clip_list, rvs_list)
    for batch_size,      bnmethod,   (mmfwd, mmgrad), opt_mm,      batch_denom,      grad_clip,          rvs in opts_gen:

        mmfwd_str = "%0.6f" % mmfwd
        mmgrad_str = "%0.6f" % mmgrad
        opt_mm_str = "%0.6f" % opt_mm
        bs = "%d" % batch_size
        shorthand = bn_shortthand_dict[bnmethod]
        if grad_clip is not None:
            grad_clip_str = "%0.3f" % grad_clip
            grad_clip_short = "_gc_%s" % grad_clip_str
        else:
            grad_clip_short = ""

        if single_bfn_mm_short is not None:
            bfn_mm_short = single_bfn_mm_short
            assert len(bfn_mm_list) == 1
        else:
            bfn_mm_short = "mmf_%s_mmg_%s" % (mmfwd_str, mmgrad_str)

        if rvs is not None:
            rvs_t, rvs_v = rvs
            assert rvs_v is None
            rvs_short = "_%s" % rvs_t
        else:
            rvs_short = ""

        if bfn_input_decay is not None:
            bi_decay_short = "_bid%0.6f" % bfn_input_decay
        else:
            bi_decay_short = ""
        shorthand_b = "%s_b%0.3d_%s_bdnm_%d_omm_%s%s%s%s" % \
            (shorthand, batch_size, bfn_mm_short, batch_denom, opt_mm_str, grad_clip_short, rvs_short, bi_decay_short)
        md = "./%s/%s_%s" % (series, dataset, shorthand_b)
        cmd_tokens = [
            'stdbuf -oL python3 %s_main.py' % dataset,
            '-dd=%s' % dd,
            '-md=%s' % md,
            '-bs=%s' % bs,
            '-bnmethod=%s' % bnmethod,
            '-mmfwd=%s' % mmfwd_str,
            '-mmgrad=%s' % mmgrad_str,
            '-opt_mm=%s' % opt_mm,
            '-batch_denom=%s' % batch_denom, ]
        if grad_clip is not None:
            cmd_tokens.append('-bfn_grad_clip=%s' % grad_clip_str)
        if resnet_size is not None:
            cmd_tokens.append('-resnet_size=%d' % resnet_size)
        if loss_scale is not None:
            cmd_tokens.append('-loss_scale=%d' % loss_scale)
        if rvs is not None:
            cmd_tokens.append('-rvst=%s' % rvs_t)
        if bfn_input_decay is not None:
            cmd_tokens.append('-bfn_input_decay=%0.6f' % bfn_input_decay)
        
        trailing_pipe = ">%s/%s_%s_%s.out 2>%s/%s_%s_%s.err" % (stderr, series, dataset, shorthand_b, \
                                                                stderr, series, dataset, shorthand_b)
        cmd_tokens.append(trailing_pipe)
        cmd = ' '.join(cmd_tokens)
        #cmd = "stdbuf -oL python3 cifar10_main.py -dd=%s -md=%s -bs=%s -bnmethod=%s -mmfwd=%s -mmgrad=%s -batch_denom=%d -opt_mm=%s %s" % \
        #    (dd, md, bs, bnmethod, mmfwd_str, mmgrad_str, batch_denom, opt_mm_str, trailing_pipe)
        print(cmd)
        md_list.append(md)


def make_cifar10_experiments_0820_b8_rvs():
    # random variance scaling. 3+ 2*5 = 13
    series = '0820_b8_rvs'
    make_cifar10_experiments_0820_helper(
        batch_size_list=(8, ),
        opt_mm_list=(.9, ),
        bnmethods_list = ('identity', 'tf_layers_regular', 'tf_layers_renorm'),
        bfn_mm_list = ((.997, .997),),
        series=series,
        batch_denom_list=(128., ),
        )
    make_cifar10_experiments_0820_helper(
        batch_size_list=(8, ),
        bnmethods_list = ('batch_free_normalization_sigfunc',),
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (.5, .5),
                       (.9, .9),
                       (.997, .997),
                      ),
        series=series,
        batch_denom_list=(128.,),
        rvs_list=(
            ('uniform_max1', None),
            ('uniform_near1', None),
            ('lognorm_max1', None),
            ('lognorm_near1', None),
            None,)
        )

def make_imagenet_experiments_0820_imagenet50():
    make_experiments_0820_helper(dataset='imagenet',
        batch_size_list=(32, ),
        bnmethods_list = ('batch_free_normalization_sigfunc', 'tf_layers_renorm', 'tf_layers_regular', 'identity'), # only tf_layers_regular matters
        bfn_mm_list=((.997, .997), ),
        opt_mm_list = (.9, ),
        batch_denom_list=(256., ),
        bfn_grad_clip_list=(100.,),
        series='imagenet50',
        resnet_size=50,
        )

def make_imagenet_experiments_0820_imagenet50redo_helper(series = 'imagenet50redo', batch_denom_list=(256., ),):
    make_experiments_0820_helper(dataset='imagenet',
        batch_size_list=(32, ),
        bnmethods_list = ('batch_free_normalization_sigfunc',),
        bfn_mm_list=(
                     (.9, .9),
                    ),
        opt_mm_list = (.9, ),
        batch_denom_list=batch_denom_list,
        series=series,
        resnet_size=50,
        bfn_grad_clip_list=(32768., ),
        loss_scale=256,
        rvs_list=(
            ('uniform_max1', None),
            ('uniform_near1', None),
            ('lognorm_max1', None),
            ('lognorm_near1', None),
            None,)
        )
def make_imagenet_experiments_0820_imagenet50redo():
    make_imagenet_experiments_0820_imagenet50redo_helper(series='imagenet50redo', batch_denom_list=(256.,))
def make_imagenet_experiments_0820_imagenet50redo2():
    make_imagenet_experiments_0820_imagenet50redo_helper(series='imagenet50redo2', batch_denom_list=(1024., ))

def make_cifar10_experiments_0820_b1_lscale():
    # using loss_scale to guard against underflow. 2*3*2+4=16 runs
    # redo some basic earlier experiments. bfn grad clip needs ot be compensated along with loss scale
    # canceled --  now using float64 accumulators
    # first experimentst to run after the tf_regular only bug was fixed.
    series = '0820_b1_lscale'
    make_cifar10_experiments_0820_helper(
        batch_size_list=(1, ),
        opt_mm_list=(.9, .99,),
        bfn_mm_list = (
                       (.5, .5),
                       (.9, .9),
                       (.997, .997),
                      ),
        series=series,
        batch_denom_list=(128.,),
        bfn_grad_clip_list=(32768., None),
        loss_scale=256,
        )
    make_cifar10_experiments_0820_helper(
        batch_size_list=(1, ),
        opt_mm_list=(.9, ),
        bnmethods_list = ('identity', 'batch_free_normalization_sigconst', 'tf_layers_regular', 'tf_layers_renorm'),
        bfn_mm_list = ((.997, .997),),
        series=series,
        batch_denom_list=(128., ),
        loss_scale=256,
        )



def make_cifar10_experiments_0821_b4():
    # 12 + 3 = 15
    # batch size 4. go back and try out sigconst again -- maybe bad perf was due to bugs earlier
    series = '0821_b4'
    make_cifar10_experiments_0820_helper(
        bnmethods_list = ('batch_free_normalization_sigconst', 'batch_free_normalization_sigfunc'),
        batch_size_list=(4., ),
        opt_mm_list=(.9, ),
        bfn_mm_list = (
                       (.5, .5),
                       (.9, .9),
                       (.9, .5),
                       (.997, .997),
                       (.997, .5),
                       (.997, .0),
                      ),
        series=series,
        batch_denom_list=(256.,),
        bfn_grad_clip_list=(32768., ),
        loss_scale=256,
        )
    make_cifar10_experiments_0820_helper(
        batch_size_list=(4., ),
        opt_mm_list=(.9, ),
        bnmethods_list = ('identity', 'tf_layers_regular', 'tf_layers_renorm'),
        bfn_mm_list = ((.997, .997),),
        series=series,
        batch_denom_list=(256., ),
        )

def make_cifar10_experiments_0821_b8_rvs_slow():
    # random variance scaling. 5 * 2 = 10
    # re-added ramping in the backend
    series = '0821_b8_rvs_slow'
    make_cifar10_experiments_0820_helper(
        batch_size_list=(8, ),
        bnmethods_list = ('batch_free_normalization_sigfunc',),
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (.5, .5),
                       (.9, .9),
                       (.9, 0.),
                       (.997, .997),
                       (.997, 0.),
                      ),
        series=series,
        bfn_grad_clip_list=(32768., ),
        batch_denom_list=(512.,),
        rvs_list=(
            # ('uniform_max1', None),
            ('uniform_near1', None),
            # ('lognorm_max1', None),
            # ('lognorm_near1', None),
            None,)
        )

def make_cifar10_experiments_0821_b8_ctrl_slow():
    # random variance scaling. 5 * 2 = 10
    series = '0821_b8_ctrl_slow'
    make_cifar10_experiments_0820_helper(
        batch_size_list=(8, ),
        bnmethods_list = ('tf_layers_regular', 'tf_layers_renorm', 'identity',),
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (.997, .997),
                      ),
        series=series,
        batch_denom_list=(512.,),
        rvs_list=(None,),
        )

def make_cifar10_experiments_0822_b8_bid2em4():
    make_cifar10_experiments_0822_b8_bid2em3(series='0822_b8_bid2em4', bfn_input_decay=2e-4)

def make_cifar10_experiments_0822_b8_bid2em5():
    make_cifar10_experiments_0822_b8_bid2em3(series='0822_b8_bid2em5', bfn_input_decay=2e-5)

def make_cifar10_experiments_0822_b8_bid2em3(series='0822_b8_bid2em3', bfn_input_decay=2e-3):
    make_cifar10_experiments_0820_helper(
        batch_size_list=(8, ),
        bnmethods_list = ('batch_free_normalization_sigfunc',),
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (.9, .9),
                       (.9, 0.),
                       (.997, .997),
                      ),
        series=series,
        bfn_grad_clip_list=(None, ),
        batch_denom_list=(128.,),
        rvs_list=(
            ('uniform_near1', None),
            None,),
        bfn_input_decay=bfn_input_decay,
        )

# cancelled -- too fast, too much bid?
def make_imagenet_experiments_0822_b64_imagenet50():
    series = '0822_b64_imagenet50'
    make_experiments_0820_helper(dataset='imagenet',
        batch_size_list=(64, ),
        bnmethods_list = ('tf_layers_renorm', 'tf_layers_regular'), # only tf_layers_regular matters
        bfn_mm_list=((.997, .997), ),
        opt_mm_list = (.9, ),
        batch_denom_list=(256., ),
        bfn_grad_clip_list=(32768.,),
        series=series,
        resnet_size=50,
        )
    make_experiments_0820_helper(dataset='imagenet',
        batch_size_list=(64, ),
        bnmethods_list = ('batch_free_normalization_sigfunc',), # only tf_layers_regular matters
        bfn_mm_list=((.9, .9), 
                     (.997, .997)),
        opt_mm_list = (.9, ),
        batch_denom_list=(256., ),
        bfn_grad_clip_list=(32768.,),
        series=series,
        resnet_size=50,
        rvs_list=(
            ('uniform_near1', None),
            None,),
        bfn_input_decay=2e-3
        )


def make_imagenet_experiments_0822_b64_imagenet50slowctrl():
    series = '0822_b64_imagenet50slowctrl'
    make_experiments_0820_helper(dataset='imagenet',
        batch_size_list=(64, ),
        bnmethods_list = ('tf_layers_renorm', 'tf_layers_regular'), # only tf_layers_regular matters
        bfn_mm_list=((.997, .997), ),
        opt_mm_list = (.9, ),
        batch_denom_list=(512., ),
        bfn_grad_clip_list=(32768.,),
        series=series,
        resnet_size=50,
        )
def make_imagenet_experiments_0822_b64_imagenet50slowbid2em6(series = '0822_b64_imagenet50slowbid2em6', bfn_input_decay=2e-6):
    make_experiments_0820_helper(dataset='imagenet',
        batch_size_list=(64, ),
        bnmethods_list = ('batch_free_normalization_sigfunc',), # only tf_layers_regular matters
        bfn_mm_list=(
                     (.5, .5), 
                     (.9, .9),  
                     #(.997, 0.)# .997 all fail. -- the mean used is simply too stale
                     ),
        opt_mm_list = (.9, ),
        batch_denom_list=(512., ),
        bfn_grad_clip_list=(None,),
        series=series,
        resnet_size=50,
        rvs_list=(None,),
        bfn_input_decay=bfn_input_decay
        )

def make_imagenet_experiments_0822_b64_imagenet50slowbid2em5():
    series = '0822_b64_imagenet50slowbid2em5'    
    make_imagenet_experiments_0822_b64_imagenet50slowbid2em6(series=series, bfn_input_decay=2e-5)


def make_cifar10_experiments_0823_b8_ctrl(series = '0823_b8_ctrl', batch_denom_list=(256., )):    
    make_cifar10_experiments_0820_helper(
        batch_size_list=(8, ),
        bnmethods_list = ('tf_layers_regular', 'tf_layers_renorm', 'identity',),
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (.997, .997),
                      ),
        series=series,
        batch_denom_list=batch_denom_list,
        rvs_list=(None,),
        )
def make_cifar10_experiments_0823_b8_ctrl128():
    make_cifar10_experiments_0823_b8_ctrl(series = '0823_b8_ctrl128', batch_denom_list=(128.,))

def make_cifar10_experiments_0823_b8_bid(series, bfn_input_decay):
    make_cifar10_experiments_0820_helper(
        batch_size_list=(8, ),
        bnmethods_list = ('batch_free_normalization_sigfunc',),
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (.5, .5),
                       (.9, .9),
                       (.9, 0.),
                       (.997, .997),
                       (.997, 0.),
                      ),
        series=series,
        bfn_grad_clip_list=(None, ),
        batch_denom_list=(256.,),
        rvs_list=(
            None,),
        bfn_input_decay=bfn_input_decay,
        )
def make_cifar10_experiments_0823_b8_bid2em456():
    make_cifar10_experiments_0823_b8_bid(series = '0823_b8_bid2em456', bfn_input_decay=2e-4)
    make_cifar10_experiments_0823_b8_bid(series = '0823_b8_bid2em456', bfn_input_decay=2e-5)
    make_cifar10_experiments_0823_b8_bid(series = '0823_b8_bid2em456', bfn_input_decay=2e-6)


def make_cifar10_experiments_0823_b8_bfnd(series, bfn_input_decay):
    make_cifar10_experiments_0820_helper(
        batch_size_list=(8, ),
        bnmethods_list = ('batch_free_direct',),
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (.9, .9),
                       (.99, .99),
                       (.997, .997),
                      ),
        series=series,
        bfn_grad_clip_list=(None, ),
        batch_denom_list=(128.,),
        rvs_list=(
            None,),
        bfn_input_decay=bfn_input_decay,
        )

def make_cifar10_experiments_0823_b8_bfnd2em3456():
    make_cifar10_experiments_0823_b8_bfnd(series = '0823_b8_bfnd2em3456', bfn_input_decay=2e-3)
    make_cifar10_experiments_0823_b8_bfnd(series = '0823_b8_bfnd2em3456', bfn_input_decay=2e-4)
    make_cifar10_experiments_0823_b8_bfnd(series = '0823_b8_bfnd2em3456', bfn_input_decay=2e-5)
    make_cifar10_experiments_0823_b8_bfnd(series = '0823_b8_bfnd2em3456', bfn_input_decay=2e-6)


def make_cifar10_experiments_0824_b8_inf(series='0824_b8_inf'):
    make_cifar10_experiments_0820_helper(
        batch_size_list=(8, ),
        bnmethods_list = ('bfn_like_regular',),
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (0., 0.),
                       (.25, .25),
                       (.5, .5),
                       (.9, .9),
                       (.25, 0.),
                       (.5, 0.),
                       (.9, 0.),
                       (.5, .25),
                       (.9, .25),
                       (.9, .5),
                      ),
        series=series,
        bfn_grad_clip_list=(None, ),
        batch_denom_list=(128.,),
        rvs_list=(None,),
        )
def make_cifar10_experiments_0825_b8_inf(series='0825_b8_inf'): # small redo, but with eps = 1e-5 like baseline, and fixed momentum
        make_cifar10_experiments_0820_helper(
        batch_size_list=(8, ),
        bnmethods_list = ('bfn_like_regular_ivtc_batch',),
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (0., 0.),
                       (0.25, 0.25),
                       (.5, .5),
                       (.9, .9),
                      ),
        series=series,
        bfn_grad_clip_list=(None, ),
        batch_denom_list=(128.,),
        rvs_list=(None,),
        )

def make_cifar10_experiments_0825_b8_loo(series='0825_b8_loo', batch_denom_list=(128.)):
    series = '0825_b8_loo'
    make_cifar10_experiments_0820_helper(
        batch_size_list=(8, ),
        bnmethods_list = ('bfn_like_regular_ivtc_batch', 
                          'bfn_like_loo_ivtc_batch',
                          'bfn_like_regular_ivtc_running',
                          'bfn_like_loo_ivtc_running',
                          'bfn_like_loo_ivtc_loo',  ## NOW DEFUNCT
                          ),
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (0., 0.),
                      ),
        series=series,
        bfn_grad_clip_list=(None, ),
        batch_denom_list=batch_denom_list,
        rvs_list=(None,),
        )

def make_cifar10_experiments_0826_b8_loof():
    series = '0826_b8_loof'
    batch_denom_list = (128.,)
    make_cifar10_experiments_0820_helper(
        batch_size_list=(8, ),
        bnmethods_list = ('bfn_like_regular_ivtc_batch', 
                          'bfn_like_loo_ivtc_batch',
                          'bfn_like_regular_ivtc_running',
                          'bfn_like_loo_ivtc_running',
                          ),
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (0., 0.),
                       (0.5, 0.5),
                      ),
        series=series,
        bfn_grad_clip_list=(None, ),
        batch_denom_list=batch_denom_list,
        rvs_list=(None,),)

if __name__ == '__main__':
    #make_cifar10_experiments_0814()
    #make_cifar10_experiments_0815()
    #make_cifar10_experiments_0815_b1()
    #make_cifar10_experiments_0816_b1()
    #make_cifar10_experiments_0816_origlike()
    #make_cifar10_experiments_0817_best_mm_gc()
    #make_cifar10_experiments_0817_resnet68()
    #make_cifar10_experiments_0817_b1()
    #make_imagenet_experiments_0817_imagenet101()
    #make_cifar10_experiments_0818_b1_slow()
    #make_cifar10_experiments_0818_b1_mmm()
    #make_cifar10_experiments_0819_b8()
    #make_cifar10_experiments_0819_b1_lscale()
    #make_cifar10_experiments_0819_b1_mmmls()
    make_cifar10_experiments_0820_b8_rvs()
    pass


