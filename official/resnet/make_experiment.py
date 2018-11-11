"""

python make_experiment.py > slurm/experiments_0829_vdw.txt  # 5 + 12 = 17 experiments on gtx
python make_experiment.py > slurm/experiments_0829_b1_vdw.txt  # 3 * 5 = 15 experiments on k80 -- increasing accuracy but also increasing ce... going to terminate early
python make_experiment.py > slurm/experiments_0829_b128_tfseq.txt  # 5 experiments. B/I/L; IBL, LBI (marginalize a lot first vs later)
python make_experiment.py > slurm/experiments_0830_b4_ctrl.txt  # 3  experiments on b4. GTX 1080 -- submitted to gtx
python make_experiment.py > slurm/experiments_0830_b4_vdw.txt  # 12  experiments on b4. GTX 1080 -- todo / to-submit onto gtx
python make_experiment.py > slurm/experiments_0913_b2_g0vdw.txt  # k80
python make_experiment.py > slurm/experiments_0830_b128_tfseq.txt  # 15-5=15 experiments. exclude: B/I/L; IBL, LBI -- queued and running on p100s

python make_experiment.py > slurm/experiments_0830_b2_ctrl.txt  # 3 experiments on k80
python make_experiment.py > slurm/experiments_0830_b2_vdw.txt  # 12 experiments on k80
python make_experiment.py > slurm/experiments_0905_b32_imagenet_tfseq.txt  # gtx 9
python make_experiment.py > slurm/experiments_0910_b128_tfseq_repro4x.txt  # 4x4 = 16 p100. (there was 4 free - run + queue)
python make_experiment.py > slurm/experiments_0909_b32_imagenet34_tfseq3x.txt  # gtx  -- cancelling -- group norm, lack of unified final affine
----------
python make_experiment.py > slurm/experiments_0913_b128_BGL.txt  # gtx 3 + 6 = 9
python make_experiment.py > slurm/experiments_0914_b32_imagenet34_BGL.txt  # gtx 3 + 6 = 9

python make_experiment.py > slurm/experiments_0916_b128_tfseq0123B56.txt  # 7 jobs on four p100 
python make_experiment.py > slurm/experiments_0917_b128_tfseq1256p.txt  # 8 jobs on four p100 -- bugs; wrong combinations... ignore. also, 1 was broken due to batch size
python make_experiment.py > slurm/experiments_0918_b128_tfseq1256p.txt  # 8 jobs on three p100 

python make_experiment.py > slurm/experiments_0919_b128_tfseq_rerun.txt # for paper use with 1 affine at the end. redoing BL experiments 5 times. 20 expeirments on 4x p100
python make_experiment.py > slurm/experiments_0919_b2_ctrl_rerun.txt  # for paper use 5x repeat of 0830_b2_ctrl -- 15 experiments on k80 -- todo -- run more of the regularization experiments?


##########
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0829_vdw.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0829_b1_vdw.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0829_b128_tfseq.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0830_b4_ctrl.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0830_b4_vdw.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0830_b128_tfseq.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0830_b2_ctrl.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0830_b2_vdw.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0905_b32_imagenet_tfseq.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0910_b128_tfseq_repro4x.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0909_b32_imagenet34_tfseq3x.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0913_b2_g0vdw.txt --src=~/gpuenv/activate

python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0913_b128_BGL.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0914_b32_imagenet34_BGL.txt --src=~/gpuenv/activate

python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0916_b128_tfseq0123B56.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0917_b128_tfseq1256p.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0918_b128_tfseq1256p.txt --src=~/gpuenv/activate

python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0919_b128_tfseq_rerun.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0919_b2_ctrl_rerun.txt --src=~/gpuenv/activate
"""



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
    'regularized_bn': 'rlsqbn',
    'regularized_bn_zero': 'rlsqbn0',
    }


def make_cifar10_experiments_0829_helper(**kwargs):
    make_experiments_0829_helper(dataset='cifar10', **kwargs)

def make_experiments_0829_helper(
        batch_size_list = (1,),
        bnmethods_list = ('batch_free_normalization_sigfunc',),
        bfn_mm_list = (
                       (.997, .997),
                       ),
        opt_mm_list = (.9, ),
        batch_denom_list=(128., ),
        bfn_grad_clip_list=(None,),
        rvs_list=(None,),
        vd_weights_list=(None,),
        series='0829',
        resnet_size=None,
        dataset='cifar10',
        loss_scale=None,
        single_bfn_mm_short=None,
        bfn_input_decay=None,
        ):
    dd="./%s_data" % dataset
    stderr='stderr'

    md_list = []
    import itertools
    opts_gen = itertools.product(
        batch_size_list, bnmethods_list, bfn_mm_list, opt_mm_list, batch_denom_list, bfn_grad_clip_list, rvs_list, vd_weights_list)
    for batch_size,      bnmethod,   (mmfwd, mmgrad), opt_mm,      batch_denom,      grad_clip,          rvs,      vd_weights in opts_gen:
        mmfwd_str = "%0.6f" % mmfwd
        mmgrad_str = "%0.6f" % mmgrad
        opt_mm_str = "%0.6f" % opt_mm
        bs = "%d" % batch_size
        shorthand = bn_shortthand_dict.get(bnmethod, None)
        if shorthand is None:
            assert bnmethod.startswith('tf_sequence')
            shorthand = 'seq' + bnmethod.split('_')[-1]
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

        if vd_weights is not None:
            vd_weight_short = "_vdw%0.2f" % vd_weights
        else:
            vd_weight_short = ""

        shorthand_tail = "".join([grad_clip_short, rvs_short, bi_decay_short, vd_weight_short])
        shorthand_b = "%s_b%0.3d_%s_bdnm_%d_omm_%s%s" % \
            (shorthand, batch_size, bfn_mm_short, batch_denom, opt_mm_str, shorthand_tail)
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
        if vd_weights is not None:
            cmd_tokens.append('-vd_weights=%0.2f' % vd_weights)
        
        trailing_pipe = ">%s/%s_%s_%s.out 2>%s/%s_%s_%s.err" % (stderr, series, dataset, shorthand_b, \
                                                                stderr, series, dataset, shorthand_b)
        cmd_tokens.append(trailing_pipe)
        cmd = ' '.join(cmd_tokens)
        #cmd = "stdbuf -oL python3 cifar10_main.py -dd=%s -md=%s -bs=%s -bnmethod=%s -mmfwd=%s -mmgrad=%s -batch_denom=%d -opt_mm=%s %s" % \
        #    (dd, md, bs, bnmethod, mmfwd_str, mmgrad_str, batch_denom, opt_mm_str, trailing_pipe)
        print(cmd)
        md_list.append(md)


def make_cifar10_experiments_0829_vdw(series = '0829_vdw', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper(
        batch_size_list=(8, ),
        bnmethods_list = ('regularized_bn',),
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (0.5, 0.5),
                       (0.9, 0.9),
                       (0.9, 0.5),
                       (0.997, 0.997),
                       (0.997, 0.5),
                      ),
        series=series,
        bfn_grad_clip_list=(None, ),
        batch_denom_list=batch_denom_list,
        rvs_list=(None,),
        )
    make_cifar10_experiments_0829_helper(
        batch_size_list=(8, ),
        bnmethods_list = ('regularized_bn',),
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (0.997, 0.997),
                       (0.99, 0.99),
                       (0.9, 0.9),
                      ),
        series=series,
        bfn_grad_clip_list=(None, ),
        batch_denom_list=batch_denom_list,
        rvs_list=(None,),
        vd_weights_list=(0., 1., 2., 4.)
        )

def make_cifar10_experiments_0829_b1_vdw(series = '0829_b1_vdw', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper(
        batch_size_list=(1, ),
        bnmethods_list = ('regularized_bn',),
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (0.997, 0.997),
                       (0.99, 0.99),
                       (0.9, 0.9),
                      ),
        series=series,
        bfn_grad_clip_list=(None, ),
        batch_denom_list=batch_denom_list,
        rvs_list=(None,),
        vd_weights_list=(0.1, 0.5, 1., 2., 4.)
        )

def make_cifar10_experiments_0829_b128_tfseq(series = '0829_b128_tfseq', batch_denom_list = (128.,)):  ##B, L paper
    make_cifar10_experiments_0829_helper(
        batch_size_list=(128, ),
        bnmethods_list = ('tf_sequence_B', 'tf_sequence_I', 'tf_sequence_L', 'tf_sequence_IBL', 'tf_sequence_LBI'),
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (0.997, 0.997),
                      ),
        series=series,
        bfn_grad_clip_list=(None, ),
        batch_denom_list=batch_denom_list,
        rvs_list=(None,),
        vd_weights_list=(None,)
        )

def make_cifar10_experiments_0830_b4_ctrl(series='0830_b4_ctrl', batch_denom_list=(128.,), batch_size_list=(4, )):
    make_cifar10_experiments_0829_helper(
        batch_size_list=batch_size_list,
        bnmethods_list = ('tf_layers_regular', 'tf_layers_renorm', 'identity'),
        opt_mm_list=(.9,),
        bfn_mm_list = ((0.997, 0.997),),
        series=series,
        batch_denom_list=batch_denom_list,
        )

def make_cifar10_experiments_0830_b4_vdw(series = '0830_b4_vdw', batch_denom_list = (128.,), 
        batch_size_list=(4, ), bnmethods_list = ('regularized_bn',)):
    make_cifar10_experiments_0829_helper(
        batch_size_list=batch_size_list,
        bnmethods_list = bnmethods_list,
        opt_mm_list=(.9,),
        bfn_mm_list = (
                       (0.9, 0.9),
                       (0.9, 0.5),
                       (0.997, 0.997),
                       (0.997, 0.5),
                      ),
        series=series,
        batch_denom_list=batch_denom_list,
        vd_weights_list=(.5, 1., 2.)
        )

def make_cifar10_experiments_0830_b2_ctrl(): ### PAPER
    make_cifar10_experiments_0830_b4_ctrl(series='0830_b2_ctrl', batch_denom_list=(128.,), batch_size_list=(2, ))
def make_cifar10_experiments_0830_b2_vdw():  ### PAPER
    make_cifar10_experiments_0830_b4_vdw(series='0830_b2_vdw', batch_denom_list=(128.,), batch_size_list=(2, ))
def make_cifar10_experiments_0913_b2_g0vdw():  ### PAPER??
    make_cifar10_experiments_0830_b4_vdw(series='0913_b2_g0vdw', batch_denom_list=(128.,), batch_size_list=(2, ), bnmethods_list = ('regularized_bn_zero', ))

#make_cifar10_experiments_0830_b2_ctrl

def make_cifar10_experiments_0830_b128_tfseq(series = '0830_b128_tfseq', batch_denom_list = (128.,)): ## Paper
    # that which was not covered in make_cifar10_experiments_0829_b128_tfseq: 15 - 5 = 10 combinations
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ('tf_sequence_BL', 'tf_sequence_LB', 'tf_sequence_BI', 'tf_sequence_IB', 'tf_sequence_LI', 'tf_sequence_IL',
                           'tf_sequence_BLI',                  'tf_sequence_BIL',                  'tf_sequence_LIB', 'tf_sequence_ILB'),
        opt_mm_list=(.9,),
        bfn_mm_list = ((0.997, 0.997),),
        series=series,
        batch_denom_list=batch_denom_list,
        )

def make_experiments_0905_b32_imagenet_tfseq(series = '0905_b32_imagenet_tfseq', batch_denom_list = (256.,)):
    # that which was not covered in make_cifar10_experiments_0829_b128_tfseq: 15 - 5 = 10 combinations
    make_experiments_0829_helper(
        dataset='imagenet',
        batch_size_list=(32, ),
        bnmethods_list =  ('tf_sequence_BL', 'tf_sequence_LB', 'tf_sequence_BI', 'tf_sequence_IB', 'tf_sequence_LI', 'tf_sequence_IL',
                           'tf_sequence_B', 'tf_sequence_L', 'tf_sequence_I',),
        opt_mm_list=(.9,),
        bfn_mm_list = ((0.997, 0.997),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=50,
        )

def make_cifar10_experiments_0910_b128_tfseq_repro(series = '0910_b128_tfseq_repro', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ('tf_sequence_BL', 'tf_sequence_LB', 'tf_sequence_B', 'tf_sequence_L', ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((0.997, 0.997),),
        series=series,
        batch_denom_list=batch_denom_list,
        )

def make_cifar10_experiments_0910_b128_tfseq_repro4x(batch_denom_list = (128.,)): # 0918: Not sure if I had more than 1 affine here... redoing.
    make_cifar10_experiments_0910_b128_tfseq_repro(series = '0910_b128_tfseq_repro1')
    make_cifar10_experiments_0910_b128_tfseq_repro(series = '0910_b128_tfseq_repro2')
    make_cifar10_experiments_0910_b128_tfseq_repro(series = '0910_b128_tfseq_repro3')
    make_cifar10_experiments_0910_b128_tfseq_repro(series = '0910_b128_tfseq_repro4')

def make_experiments_0909_b32_imagenet34_tfseq(series = '0910_b32_imagenet34_tfseq', batch_denom_list = (256.,)):
    # that which was not covered in make_cifar10_experiments_0829_b128_tfseq: 15 - 5 = 10 combinations
    make_experiments_0829_helper(
        dataset='imagenet',
        batch_size_list=(32, ),
        bnmethods_list =  ('tf_sequence_BL', 'tf_sequence_LB', 'tf_sequence_L', 'tf_sequence_B'),
        opt_mm_list=(.9,),
        bfn_mm_list = ((0.997, 0.997),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=34,
        )
def make_experiments_0909_b32_imagenet34_tfseq3x(batch_denom_list = (256.,)): # use to offset the dates a bit...
    make_experiments_0909_b32_imagenet34_tfseq(series='0909_b32_imagenet34_tfseq1')
    make_experiments_0909_b32_imagenet34_tfseq(series='0909_b32_imagenet34_tfseq2')
    make_experiments_0909_b32_imagenet34_tfseq(series='0909_b32_imagenet34_tfseq3')


def make_cifar10_experiments_0913_b128_BGL(series='0913_b128_BGL', batch_denom_list = (128.,)): # with group norm. also, unified affine to be after everything
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ('tf_sequence_BG', 'tf_sequence_BL', 'tf_sequence_GB', 'tf_sequence_GL', 'tf_sequence_LB', 'tf_sequence_LG',
                'tf_sequence_B', 'tf_sequence_G', 'tf_sequence_L', ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((0.997, 0.997),),
        series=series,
        batch_denom_list=batch_denom_list,
    )

def make_imagenet_experiments_0914_b32_BGL(series='0914_b132_imagenet34_BGL', batch_denom_list = (256.,)): # with group norm. also, unified affine to be after everything
    make_experiments_0829_helper(
        dataset='imagenet',
        batch_size_list=(32, ),
        bnmethods_list =  ('tf_sequence_BG', 'tf_sequence_BL', 'tf_sequence_GB', 'tf_sequence_GL', 'tf_sequence_LB', 'tf_sequence_LG',
                'tf_sequence_B', 'tf_sequence_G', 'tf_sequence_L', ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((0.997, 0.997),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=34,
        )


def make_cifar10_experiments_0916_b128_tfseq0123B56(series = '0916_b128_tfseq0123B56', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ('tf_sequence_0', 'tf_sequence_1', 'tf_sequence_2', 'tf_sequence_3', 'tf_sequence_B', 'tf_sequence_5', 'tf_sequence_6'),
        opt_mm_list=(.9,),
        bfn_mm_list = ((0.997, 0.997),),
        series=series,
        batch_denom_list=batch_denom_list,
        )

def make_cifar10_experiments_0917_b128_tfseq1256p(series = '0917_b128_tfseq1256p', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ('tf_sequence_12', 'tf_sequence_16', 'tf_sequence_23', 'tf_sequence_25',
                           'tf_sequence_21', 'tf_sequence_61', 'tf_sequence_32', 'tf_sequence_52'),
        opt_mm_list=(.9,),
        bfn_mm_list = ((0.997, 0.997),),
        series=series,
        batch_denom_list=batch_denom_list,
        ) # wrong sequences; also 1 and 0 were broken
 
def make_cifar10_experiments_0918_b128_tfseq1256p(series = '0918_b128_tfseq1256p', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ('tf_sequence_12', 'tf_sequence_16', 'tf_sequence_52', 'tf_sequence_56', # accumulate W, then H
                           'tf_sequence_21', 'tf_sequence_61', 'tf_sequence_52', 'tf_sequence_65'), # reverse order
        opt_mm_list=(.9,),
        bfn_mm_list = ((0.997, 0.997),),
        series=series,
        batch_denom_list=batch_denom_list,
        )


def make_cifar10_experiments_0919_b128_tfseq_rerun(batch_denom_list = (128.,)): # 0918: Not sure if I had more than 1 affine here... redoing.
    make_cifar10_experiments_0910_b128_tfseq_repro(series = '0919_b128_tfseq_BL_rerun1')
    make_cifar10_experiments_0910_b128_tfseq_repro(series = '0919_b128_tfseq_BL_rerun2')
    make_cifar10_experiments_0910_b128_tfseq_repro(series = '0919_b128_tfseq_BL_rerun3')
    make_cifar10_experiments_0910_b128_tfseq_repro(series = '0919_b128_tfseq_BL_rerun4')
    make_cifar10_experiments_0910_b128_tfseq_repro(series = '0919_b128_tfseq_BL_rerun5')

#experiments_0919_b2_ctrl_rerun


def make_cifar10_experiments_0919_b2_ctrl_rerun(batch_denom_list = (128.,)): # 0918: Not sure if I had more than 1 affine here... redoing.
    make_cifar10_experiments_0830_b4_ctrl(series='0919_b2_ctrl_rerun1', batch_denom_list=(128.,), batch_size_list=(2, ))
    make_cifar10_experiments_0830_b4_ctrl(series='0919_b2_ctrl_rerun2', batch_denom_list=(128.,), batch_size_list=(2, ))
    make_cifar10_experiments_0830_b4_ctrl(series='0919_b2_ctrl_rerun3', batch_denom_list=(128.,), batch_size_list=(2, ))
    make_cifar10_experiments_0830_b4_ctrl(series='0919_b2_ctrl_rerun4', batch_denom_list=(128.,), batch_size_list=(2, ))
    make_cifar10_experiments_0830_b4_ctrl(series='0919_b2_ctrl_rerun5', batch_denom_list=(128.,), batch_size_list=(2, ))


if __name__ == '__main__':
    #make_cifar10_experiments_0825_b8_loo_d512()
    #make_cifar10_experiments_0825_b8_inf()
    #make_cifar10_experiments_0829_vdw() # was b8
    #make_cifar10_experiments_0829_b1_vdw()
    #make_cifar10_experiments_0829_b128_tfseq()
    #make_cifar10_experiments_0830_b4_ctrl()
    #make_cifar10_experiments_0830_b4_vdw()
    #make_cifar10_experiments_0830_b2_ctrl()
    #make_cifar10_experiments_0830_b2_vdw()
    #make_cifar10_experiments_0830_b128_tfseq()
    #make_experiments_0905_b32_imagenet_tfseq()
    #make_cifar10_experiments_0910_b128_tfseq_repro4x()
    #make_experiments_0909_b32_imagenet34_tfseq3x()
    #make_cifar10_experiments_0913_b2_g0vdw()
    #make_cifar10_experiments_0913_b128_BGL()
    #make_imagenet_experiments_0914_b32_BGL()

    #make_cifar10_experiments_0916_b128_tfseq0123B56()
    #make_cifar10_experiments_0917_b128_tfseq1256p()
    #make_cifar10_experiments_0918_b128_tfseq1256p()
    
    #make_cifar10_experiments_0919_b128_tfseq_rerun()
    make_cifar10_experiments_0919_b2_ctrl_rerun()
    pass
