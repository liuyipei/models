"""

python make_experiment.py > slurm/experiments_0919_b128_tfseq_rerun.txt # for paper use with 1 affine at the end. redoing BL experiments 5 times. 20 expeirments on 4x p100
python make_experiment.py > slurm/experiments_0919_b2_ctrl_rerun.txt  # for paper use 5x repeat of 0830_b2_ctrl -- 15 experiments on k80 -- todo -- run more of the regularization experiments?
ICLR SUBMIT


python3 make_experiment.py > slurm/experiments_1111_b128_lsqrn.txt 
python3 make_experiment.py > slurm/experiments_1112_lsqrn_eps.txt # 1 + (7 + 5 + 3 + 1) = 17
python3 make_experiment.py > slurm/experiments_1113_lsqrn_afterall.txt # 1 + (7 + 5 + 3 + 1) = 17
python3 make_experiment.py > slurm/1113_lsqrn_imagenet.txt # 1 + 1 + (2 * 5) + 1 = 13
python3 make_experiment.py > slurm/1114_lsqrn_norm1.txt
python3 make_experiment.py > slurm/1114_bn_baseline.txt
python3 make_experiment.py > slurm/1115_lsqrn_fast.txt
python3 make_experiment.py > slurm/1116_lsqrn_marg.txt
python3 make_experiment.py > slurm/1117_lsqrn_switch.txt
python3 make_experiment.py > slurm/1118_imn18sw.txt
python3 make_experiment.py > slurm/1119_imn18baseline.txt # 10 runs
python3 make_experiment.py > slurm/1120_shared.txt       # 4 + 3 7
python3 make_experiment.py > slurm/1121_im18shared.txt       # 4 + 3 7
python3 make_experiment.py > slurm/1121_sharedmHW.txt       # 4
python3 make_experiment.py > slurm/1121_im18sharedmHW.txt       # 5
python3 make_experiment.py > slurm/1122_shareduse1g.txt       # 4
python3 make_experiment.py > slurm/1122_im18shareduse1g.txt       # 5
python3 make_experiment.py > slurm/1124_switch_baseline.txt       # 5
python3 make_experiment.py > slurm/1125_im18_switch_baseline.txt       # 4
python3 make_experiment.py > slurm/1125_im34_switch_baseline.txt       # 4
python3 make_experiment.py > slurm/1125_im34_shareduse1gmHW.txt        $ 5
python3 make_experiment.py > slurm/1126_im34_l2tr.txt # 10    #### IMPORTANT -- 12tr is important especially to marg-BHW
python3 make_experiment.py > slurm/1129_im34_bothmarg.txt # 4 
python3 make_experiment.py > slurm/1130_im18_l2tr_1g2g.txt # 10
python3 make_experiment.py > slurm/1131_im50_switch_baseline.txt # 4 -- cancelled -- restarted on 1209
python3 make_experiment.py > slurm/1132_im18_l2tr_fullgroup.txt # 10

python3 make_experiment.py > slurm/1201_im50_BHW_both.txt # 12

##########

python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0919_b128_tfseq_rerun.txt --src=~/gpuenv/activate
python ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_0919_b2_ctrl_rerun.txt --src=~/gpuenv/activate
ICLR SUBMIT

python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_1111_b128_lsqrn.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_1112_lsqrn_eps.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/experiments_1113_lsqrn_afterall.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/make_cifar10_experiments_1113_lsqrn_imagenet.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/make_cifar10_experiments_1114_lsqrn_norm1.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1114_bn_baseline.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1115_lsqrn_fast.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1117_lsqrn_switch.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1118_imn18sw.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1119_imn18baseline.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1120_shared.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1121_im18shared.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1121_sharedmHW.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1121_im18sharedmHW.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1122_shareduse1g.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1122_im18shareduse1g.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1124_switch_baseline.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1125_im18_switch_baseline.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1125_im34_switch_baseline.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1125_im34_shareduse1gmHW.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1126_im34_l2tr.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1129_im34_bothmarg.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/temp_bothmarg.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1130_im18_l2tr_1g2g.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1131_im50_switch_baseline.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1132_im18_l2tr_fullgroup.txt --src=~/gpuenv/activate
python3 ~/jobscripts/general/general_sbatch2.py --wd=/home/yiliu/gitmisc/external/models/official/resnet --cmdlist=slurm/1201_im50_BHW_both.txt --src=~/gpuenv/activate

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

def make_imagenet_experiments_0829_helper(**kwargs):
    make_experiments_0829_helper(dataset='imagenet', **kwargs)

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
        append_int_id=False,
        ):
    dd="./%s_data" % dataset
    stderr='stderr'

    import itertools
    opts_gen = itertools.product(
             batch_size_list, bnmethods_list, bfn_mm_list, opt_mm_list, batch_denom_list, bfn_grad_clip_list, rvs_list, vd_weights_list)
    for _i, (batch_size,      bnmethod,   (mmfwd, mmgrad), opt_mm,      batch_denom,      grad_clip,          rvs,      vd_weights) \
            in enumerate(opts_gen):
        mmfwd_str = None if mmfwd is None else "%0.6f" % mmfwd
        mmgrad_str = None if mmgrad is None else "%0.6f" % mmgrad
        opt_mm_str = "%0.6f" % opt_mm
        bs = "%d" % batch_size
        shorthand = bn_shortthand_dict.get(bnmethod, None)
        if shorthand is None:
            if bnmethod.startswith('tf_sequence'):
                shorthand = 'seq' + bnmethod.split('_')[-1]
            elif bnmethod.startswith('lsqrn') or bnmethod.startswith('sharedlsqrn'):
                _bnm_prefix = (bnmethod.split('_')[0]) + '_'
                shorthand = _bnm_prefix + "_".join(bnmethod.split('_')[1:])
            elif bnmethod.startswith('switch'):
                shorthand = bnmethod
            else:
                raise ValueError()
        if grad_clip is not None:
            grad_clip_str = "%0.3f" % grad_clip
            grad_clip_short = "_gc_%s" % grad_clip_str
        else:
            grad_clip_short = ""

        bfn_mm_short = ""
        if single_bfn_mm_short is not None:
            bfn_mm_short = single_bfn_mm_short
            assert len(bfn_mm_list) == 1
        else:
            if mmfwd is not None or mmgrad is not None:
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
        if append_int_id:
            shorthand_b += '_id%0.2d' % _i
        md = "./%s/%s_%s" % (series, dataset, shorthand_b) # MODEL DIR

        cmd_tokens = [
            'stdbuf -oL python3 %s_main.py' % dataset,
            '-dd=%s' % dd,
            '-md=%s' % md,
            '-bs=%s' % bs,
            '-bnmethod=%s' % bnmethod,
            '-opt_mm=%s' % opt_mm,
            '-batch_denom=%s' % batch_denom, 
            '-data_format=%s' % 'channels_last']
        if mmfwd is not None:
            cmd_tokens.append('-mmfwd=%s' % mmfwd_str)
        if mmgrad is not None:
            cmd_tokens.append('-mmgrad=%s' % mmgrad_str)
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

def make_cifar10_experiments_1111_b128_lsqrn(series='1111_b128_lsqrn', batch_denom_list = (128.,)): # with group norm. also, unified affine to be after everything
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ( # lsqrn_c_I.
                'tf_layers_regular',
                'lsqrn_8_4',
                'lsqrn_4_2', 
                'lsqrn_2_1',
                ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((0.997, 0.997),),
        series=series,
        batch_denom_list=batch_denom_list,
    )

def make_cifar10_experiments_1112_lsqrn_eps(series='1112_b128_lsqrn_eps', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ( # lsqrn_c_I.
                'tf_layers_regular',
                'lsqrn_16_15', 'lsqrn_16_14', 'lsqrn_16_12', 'lsqrn_16_8', 'lsqrn_16_4', 'lsqrn_16_2', 'lsqrn_16_1',
                'lsqrn_8_7', 'lsqrn_8_6', 'lsqrn_8_4', 'lsqrn_8_2', 'lsqrn_8_1',
                'lsqrn_4_3', 'lsqrn_4_2', 'lsqrn_4_1',
                'lsqrn_2_1',
                ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
    )

def make_cifar10_experiments_1113_lsqrn_afterall(series='1113_b128_lsqrn_afterall', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ( # lsqrn_c_I.
                'tf_layers_regular',
                'lsqrn_16_15', 'lsqrn_16_14', 'lsqrn_16_12', 'lsqrn_16_8', 'lsqrn_16_4', 'lsqrn_16_2', 'lsqrn_16_1',
                'lsqrn_8_7', 'lsqrn_8_6', 'lsqrn_8_4', 'lsqrn_8_2', 'lsqrn_8_1',
                'lsqrn_4_3', 'lsqrn_4_2', 'lsqrn_4_1',
                'lsqrn_2_1',
                ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
    )

def make_cifar10_experiments_1113_lsqrn_imagenet(series='1113_b32_lsqrn_imagenet', batch_denom_list = (256.,)):
    make_imagenet_experiments_0829_helper( 
        batch_size_list=(32, ),
        bnmethods_list =  ( # lsqrn_c_I.
                'tf_layers_regular',
                'tf_layers_renorm',
                'lsqrn_64_32', 'lsqrn_64_1',
                'lsqrn_32_16', 'lsqrn_32_1',
                'lsqrn_16_8',  'lsqrn_16_1',
                'lsqrn_8_4',   'lsqrn_8_1',
                'lsqrn_4_2',   'lsqrn_4_1',
                'lsqrn_2_1',
                ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
    )


def make_cifar10_experiments_1114_lsqrn_norm1(series='1114_lsqrn_norm1', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ( # lsqrn_c_I.
                'tf_layers_regular',
                'lsqrn_16_15', 'lsqrn_16_14', 'lsqrn_16_8', 'lsqrn_16_2', 'lsqrn_16_1',
                'lsqrn_8_7', 'lsqrn_8_6', 'lsqrn_8_4', 'lsqrn_8_2', 'lsqrn_8_1',
                'lsqrn_4_3', 'lsqrn_4_2', 'lsqrn_4_1',
                'lsqrn_2_1',
                #'lsqrnfast_16_15', 'lsqrnfast_16_14', 'lsqrnfast_16_8', 'lsqrnfast_16_2', 'lsqrnfast_16_1',
                #'lsqrnfast_8_7', 'lsqrnfast_8_6', 'lsqrnfast_8_4', 'lsqrnfast_8_2', 'lsqrnfast_8_1',
                #'lsqrnfast_4_3', 'lsqrnfast_4_2', 'lsqrnfast_4_1',
                #'lsqrnfast_2_1',
                ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
    )


def make_cifar10_experiments_1114_bn_baseline(series='1114_lsqrn_bn_baseline', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ( # lsqrn_c_I.
                'tf_layers_regular', 'tf_layers_regular', 'tf_layers_regular', 'tf_layers_regular', 'tf_layers_regular', 
                'tf_layers_regular', 'tf_layers_regular', 'tf_layers_regular', 'tf_layers_regular', 'tf_layers_regular', 
                ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
        append_int_id=True,
    )

def make_cifar10_experiments_1115_lsqrn_fast(series='1115_lsqrn_fast', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ( # lsqrn_c_I.
                'lsqrnfast_16_15', 'lsqrnfast_16_14', 'lsqrnfast_16_8', 'lsqrnfast_16_2', 'lsqrnfast_16_1',
                'lsqrnfast_8_7', 'lsqrnfast_8_6', 'lsqrnfast_8_4', 'lsqrnfast_8_2', 'lsqrnfast_8_1',
                'lsqrnfast_4_3', 'lsqrnfast_4_2', 'lsqrnfast_4_1',
                'lsqrnfast_2_1',
                'lsqrn_16_15', 'lsqrn_16_14', 'lsqrn_16_8', 'lsqrn_16_2', 'lsqrn_16_1',
                'lsqrn_8_7', 'lsqrn_8_6', 'lsqrn_8_4', 'lsqrn_8_2', 'lsqrn_8_1',
                'lsqrn_4_3', 'lsqrn_4_2', 'lsqrn_4_1',
                'lsqrn_2_1',
                ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
    )

def make_cifar10_experiments_1116_lsqrn_marg(series='1116_lsqrn_marg', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ( # lsqrn_c_I.
            'lsqrnfast-mHW_16_15', 'lsqrnfast-mHW_16_14', 'lsqrnfast-mHW_16_8', 'lsqrnfast-mHW_16_2', 'lsqrnfast-mHW_16_1',
            'lsqrnfast-mHW_8_7', 'lsqrnfast-mHW_8_6', 'lsqrnfast-mHW_8_4', 'lsqrnfast-mHW_8_2', 'lsqrnfast-mHW_8_1',
            'lsqrnfast-mHW_4_3', 'lsqrnfast-mHW_4_2', 'lsqrnfast-mHW_4_1',
            'lsqrnfast-mHW_2_1',
            'lsqrn-mHW_16_15', 'lsqrn-mHW_16_14', 'lsqrn-mHW_16_8', 'lsqrn-mHW_16_2', 'lsqrn-mHW_16_1',
            'lsqrn-mHW_8_7', 'lsqrn-mHW_8_6', 'lsqrn-mHW_8_4', 'lsqrn-mHW_8_2', 'lsqrn-mHW_8_1',
            'lsqrn-mHW_4_3', 'lsqrn-mHW_4_2', 'lsqrn-mHW_4_1',
            'lsqrn-mHW_2_1',
                ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
    )

def make_cifar10_experiments_1117_switch(series='1117_switch', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ( # lsqrn_c_I.
            'switch',              'switch-fast',             'switch-block2',
            'switch-lsqrn_16_8',   'switch-fast-lsqrn_16_8',  'switch-block2-lsqrn_16_8',
            'switch-lsqrn_16_14',  'switch-fast-lsqrn_16_14', 'switch-block2-lsqrn_16_14',
            'lsqrn_16_8',     'lsqrnfast_16_8',     'lsqrnblock2_16_8',
            'lsqrn_16_14',    'lsqrnfast_16_14',    'lsqrnblock2_16_14',
            'lsqrn-mHW_16_8', 'lsqrnfast-mHW_16_8', 'lsqrnblock2-mHW_16_8',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
    )

def make_imagenet18_experiments_1118_imn18sw(series='1118_imn18sw', batch_denom_list = (256.,)):
    make_imagenet_experiments_0829_helper( 
        batch_size_list=(32, ),
        bnmethods_list =  ( # lsqrn_c_I.
            'switch',              'switch-fast',             'switch-block2',
            'switch-lsqrn_16_8',   'switch-fast-lsqrn_16_8',  'switch-block2-lsqrn_16_8',
            'switch-lsqrn_16_14',  'switch-fast-lsqrn_16_14', 'switch-block2-lsqrn_16_14',
            'lsqrn_16_8',     'lsqrnfast_16_8',     'lsqrnblock2_16_8',
            'lsqrn_16_14',    'lsqrnfast_16_14',    'lsqrnblock2_16_14',
            'lsqrn-mHW_16_8', 'lsqrnfast-mHW_16_8', 'lsqrnblock2-mHW_16_8',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=18,
    )


def make_imagenet18_experiments_1119_imn18baseline(series='1119_imn18baseline', batch_denom_list = (256.,)):
    make_imagenet_experiments_0829_helper( 
        batch_size_list=(32, ),
        bnmethods_list =  ( # lsqrn_c_I.
            'tf_layers_regular', 'tf_layers_regular', 'tf_layers_regular', 'tf_layers_regular', 'tf_layers_regular', 
            'tf_layers_regular', 'tf_layers_regular', 'tf_layers_regular', 'tf_layers_regular', 'tf_layers_regular', 
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=18,
        append_int_id=True,
    )


def make_cifar10_experiments_1120_shared(series='1120_shared', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ( # lsqrn_c_I.
            'sharedlsqrn_16_0', 'sharedlsqrn_8_0', 'sharedlsqrn_4_0', 'sharedlsqrn_2_0',
            'switch',            'switch-fast',    'switch-block2',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
    )

def make_imagenet18_experiments_1121_im18shared(series='1121_im18shared', batch_denom_list = (256.,)):
    make_imagenet_experiments_0829_helper( 
        batch_size_list=(32, ),
        bnmethods_list =  ( # lsqrn_c_I.
            'sharedlsqrn_16_0', 'sharedlsqrn_8_0', 'sharedlsqrn_4_0', 'sharedlsqrn_2_0',
            'switch',            'switch-fast',    'switch-block2',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=18,
    )

def make_cifar10_experiments_1121_sharedmHW(series='1121_sharedmHW', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ( # lsqrn_c_I.
            'sharedlsqrn-mHW_16_0', 'sharedlsqrn-mHW_8_0', 'sharedlsqrn-mHW_4_0', 'sharedlsqrn-mHW_2_0',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
    )
def make_cifar10_experiments_1121_im18sharedmHW(series='1121_im18sharedmHW', batch_denom_list = (256.,)):
    make_imagenet_experiments_0829_helper( 
        batch_size_list=(32, ),
        bnmethods_list =  ( # lsqrn_c_I.
             'sharedlsqrn-mHW_32_0', 'sharedlsqrn-mHW_16_0', 'sharedlsqrn-mHW_8_0', 'sharedlsqrn-mHW_4_0', 'sharedlsqrn-mHW_2_0',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=18,
    )




def make_cifar10_experiments_1122_shareduse1g(series='1122_shareduse1g', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ( # lsqrn_c_I.
            'sharedlsqrn-use1g_16_0', 'sharedlsqrn-use1g_8_0', 'sharedlsqrn-use1g_4_0', 'sharedlsqrn-use1g_2_0',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
    )
def make_experiments_1122_im18shareduse1g(series='1122_im18shareduse1g', batch_denom_list = (256.,)):
    make_imagenet_experiments_0829_helper( 
        batch_size_list=(32, ),
        bnmethods_list =  ( # lsqrn_c_I.
             'sharedlsqrn-use1g_32_0', 'sharedlsqrn-use1g_16_0', 'sharedlsqrn-use1g_8_0', 'sharedlsqrn-use1g_4_0', 'sharedlsqrn-use1g_2_0',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=18,
    )


def make_cifar10_experiments_1124_switch_baseline(series='1124_switch_baseline', batch_denom_list = (128.,)):
    make_cifar10_experiments_0829_helper( 
        batch_size_list=(128, ),
        bnmethods_list =  ( # lsqrn_c_I.
            'switch', 'switch', 'switch', 'switch', 'switch',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
        append_int_id=True,
    )
def make_experiments_1125_im18_switch_baseline(series='1125_im18_switch_baseline', batch_denom_list = (256.,)):
    make_imagenet_experiments_0829_helper( 
        batch_size_list=(32, ),
        bnmethods_list =  ( # lsqrn_c_I.
             'tf_layers_regular', 'tf_layers_regular',
             'switch', 'switch',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=18,
        append_int_id=True
    )


      #34: [3, 4, 6, 3],
      #50: [3, 4, 6, 3],
      #101: [3, 4, 23, 3],
#### iclr reject

def make_experiments_1125_im34_switch_baseline(series='1125_im34_switch_baseline', batch_denom_list = (256.,)):
    make_imagenet_experiments_0829_helper( 
        batch_size_list=(32, ),
        bnmethods_list =  ( # lsqrn_c_I.
             'tf_layers_regular', 'tf_layers_regular',
             'switch', 'switch',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=34,
        append_int_id=True
    )

def make_experiments_1125_im34_shareduse1gmHW(series='1125_im34_shareduse1gmHW', batch_denom_list = (256.,)):
    make_imagenet_experiments_0829_helper( 
        batch_size_list=(32, ),
        bnmethods_list =  ( # lsqrn_c_I.
             'sharedlsqrn-use1g_32_0', 'sharedlsqrn-use1g_16_0', 'sharedlsqrn-use1g_8_0',
             'sharedlsqrn-use1g_4_0', 'sharedlsqrn-use1g_2_0',
             'sharedlsqrn-use1g-mHW_32_0', 'sharedlsqrn-use1g-mHW_16_0', 'sharedlsqrn-use1g-mHW_8_0', 
             'sharedlsqrn-use1g-mHW_4_0', 'sharedlsqrn-use1g-mHW_2_0',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=34,
    )



def make_experiments_1126_im34_l2tr(series='1126_im34_l2tr', batch_denom_list = (256.,)):
    make_imagenet_experiments_0829_helper( 
        batch_size_list=(32, ),
        bnmethods_list =  ( # lsqrn_c_I.
             'sharedlsqrn-use1g-l2tr_32_0', 'sharedlsqrn-use1g-l2tr_16_0', 'sharedlsqrn-use1g-l2tr_8_0',
             'sharedlsqrn-use1g-l2tr_4_0', 'sharedlsqrn-use1g-l2tr_2_0',
             'sharedlsqrn-use1g-mHW-l2tr_32_0', 'sharedlsqrn-use1g-mHW-l2tr_16_0', 'sharedlsqrn-use1g-mHW-l2tr_8_0', 
             'sharedlsqrn-use1g-mHW-l2tr_4_0', 'sharedlsqrn-use1g-mHW-l2tr_2_0',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=34,
    )


def make_experiments_1129_im34_bothmarg(series='1129_im34_bothmarg', batch_denom_list = (256.,)):
    make_imagenet_experiments_0829_helper( 
        batch_size_list=(32, ),
        bnmethods_list =  ( # lsqrn_c_I.
             'sharedlsqrn-1g.per.HW.BHW-l2tr_16_0', 'sharedlsqrn-1g.per.HW.BHW-l2tr_8_0', 
             'sharedlsqrn-1g.per.HW.BHW-l2tr_4_0', 'sharedlsqrn-1g.per.HW.BHW-l2tr_2_0',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=34,
    )


def make_experiments_1130_im18_l2tr_1g2g(series='1130_im18_l2tr_1g2g', batch_denom_list = (256.,)):
    make_imagenet_experiments_0829_helper( 
        batch_size_list=(32, ),
        bnmethods_list =  ( # lsqrn_c_I.
             'sharedlsqrn-use1g-l2tr_16_0',         'sharedlsqrn-use1g-l2tr_8_0',         'sharedlsqrn-use1g-l2tr_4_0', 
             'sharedlsqrn-use1g-mHW-l2tr_16_0',     'sharedlsqrn-use1g-mHW-l2tr_8_0',     'sharedlsqrn-use1g-mHW-l2tr_4_0', 
             'sharedlsqrn-1g.per.HW.BHW-l2tr_16_0', 'sharedlsqrn-1g.per.HW.BHW-l2tr_8_0', 
             'sharedlsqrn-1g.per.HW.BHW-l2tr_4_0',  'sharedlsqrn-1g.per.HW.BHW-l2tr_2_0',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=18,
    )

def make_experiments_1131_im50_switch_baseline(series='1131_im50_switch_baseline', batch_denom_list = (256.,)):
    make_imagenet_experiments_0829_helper( 
        batch_size_list=(32, ),
        bnmethods_list =  ( # lsqrn_c_I.
             'tf_layers_regular', 'tf_layers_regular',
             'switch', 'switch',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=50,
        append_int_id=True
    )


def make_experiments_1132_im18_l2tr_fullgroup(series='1132_im18_l2tr_fullgroup', batch_denom_list = (256.,)):
    make_imagenet_experiments_0829_helper( 
        batch_size_list=(32, ),
        bnmethods_list =  ( # lsqrn_c_I.
             'sharedlsqrn-l2tr_16_0',         'sharedlsqrn-l2tr_8_0',         'sharedlsqrn-l2tr_4_0', 
             'sharedlsqrn-mHW-l2tr_16_0',     'sharedlsqrn-mHW-l2tr_8_0',     'sharedlsqrn-mHW-l2tr_4_0', 
             'sharedlsqrn-fullgroup.per.HW.BHW-l2tr_16_0', 'sharedlsqrn-fullgroup.per.HW.BHW-l2tr_8_0', 
             'sharedlsqrn-fullgroup.per.HW.BHW-l2tr_4_0',  'sharedlsqrn-fullgroup.per.HW.BHW-l2tr_2_0',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=18,
    )


def make_experiments_1201_im50_BHW_both(series='1201_im50_BHW_both', batch_denom_list = (256.,)):
    # does not have the mHW only experiments since they are likely to have bad perf
    make_imagenet_experiments_0829_helper( 
        batch_size_list=(32, ),
        bnmethods_list =  ( # lsqrn_c_I.
             'sharedlsqrn-l2tr_8_0',         'sharedlsqrn-l2tr_4_0', 
             'sharedlsqrn-use1g-l2tr_8_0',         'sharedlsqrn-use1g-l2tr_4_0', 
             'sharedlsqrn-fullgroup.per.HW.BHW-l2tr_16_0', 'sharedlsqrn-fullgroup.per.HW.BHW-l2tr_8_0',
                'sharedlsqrn-fullgroup.per.HW.BHW-l2tr_4_0', 'sharedlsqrn-fullgroup.per.HW.BHW-l2tr_2_0',
             'sharedlsqrn-1g.per.HW.BHW-l2tr_16_0',  'sharedlsqrn-1g.per.HW.BHW-l2tr_8_0',
                'sharedlsqrn-1g.per.HW.BHW-l2tr_4_0', 'sharedlsqrn-1g.per.HW.BHW-l2tr_2_0',
        ),
        opt_mm_list=(.9,),
        bfn_mm_list = ((None, None),),
        series=series,
        batch_denom_list=batch_denom_list,
        resnet_size=50,
    )

if __name__ == '__main__':
    # make_cifar10_experiments_0919_b128_tfseq_rerun()
    # make_cifar10_experiments_0919_b2_ctrl_rerun()
    # make_cifar10_experiments_1111_b128_lsqrn()
    # make_cifar10_experiments_1112_lsqrn_eps()
    # make_cifar10_experiments_1113_lsqrn_afterall()
    # make_cifar10_experiments_1114_bn_baseline()
    # make_cifar10_experiments_1115_lsqrn_fast()
    # make_cifar10_experiments_1116_lsqrn_marg()
    # make_cifar10_experiments_1117_switch()
    # make_imagenet18_experiments_1118_imn18sw()
    # make_imagenet18_experiments_1119_imn18baseline()
    # make_cifar10_experiments_1120_shared()
    # make_imagenet18_experiments_1121_im18shared()
    # make_cifar10_experiments_1121_sharedmHW
    # make_cifar10_experiments_1121_im18sharedmHW()
    # make_experiments_1122_im18sharedmHW()

    # make_cifar10_experiments_1122_shareduse1g()
    # make_experiments_1122_im18shareduse1g()
    # make_cifar10_experiments_1124_switch_baseline()
    # make_experiments_1125_im18_switch_baseline()
    # make_experiments_1125_im34_switch_baseline()
    # make_experiments_1125_im34_shareduse1gmHW()

    # make_experiments_1126_im34_l2tr()
    # make_experiments_1129_im34_bothmarg()
    # make_experiments_1130_im18_l2tr_1g2g()
    # make_experiments_1131_im50_switch_baseline()
    # make_experiments_1132_im18_l2tr_fullgroup()

    make_experiments_1201_im50_BHW_both()

    # pass

