import numpy as np

bn_shortthand_dict = {
    'tf_layers_regular': 'baselinebn', 
    'tf_layers_renorm': 'renorm', 
    'batch_free_normalization_sigfunc': 'bfsigf',
    'batch_free_normalization_sigconst': 'bfsigc',
    'batch_free_normalization_sigfunc_compare_running_stats': 'bfsigfcmp',
    'identity': 'identity'}


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
            rvs_short = ""

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
