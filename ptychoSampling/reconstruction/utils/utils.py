import tensorflow as tf
from ptychoSampling.logger import logger
from typing import Union, List

def getComputationalCostInFlops(graph: tf.Graph,
                                keywords: List[Union[str, List]] = None,
                                exclude_keywords: bool = True) -> int:
    """Calculate the number of flops required to run the graph, while excluding any nodes that contain the supplied
    keywords.

    Parameters
    ----------
    graph : tf.Graph
        Tensorflow graph object to analyze.
    keywords : list(str or list)
        Select the nodes containing the specified keywords in the node names. An individual "keyword" item can be either a
        string, or a list of strings. When a list of strings is supplied as a keyword, the node is selected using a
        logical and operator, i.e., all the strings must be present in the node name. When `exclude_keywords` is
        set to `True`, the total flops returned excludes the flops used for the specified nodes.
        When `exclude_keywords` is set to `False`, the return value only contains the total flops used for the
        specified keywords. Default value of `None` returns the total flops for the entire computational graph.
    exclude_keywords : bool
        Whether to exclude (or include only) the specified keywords from the total flops counted.
    Returns
    -------
    flops : int
        Total number of flops required for one pass through the graph.
    Notes
    -----
    Calculates the flops using the estimates from `sopt.benchmarks.ops.tensorflow.flops_registry_custom`. The
    estimates used might not include all the operations used in the supplied graph.

    I went carefully through the nodes for the simple case of ePIE reconstruction, where I found that this function
    did actually correctly include all the relevant nodes.
    """

    from tensorflow.python.framework import graph_util
    import sopt.benchmarks.ops.tensorflow.flops_registry_custom
    from sopt.benchmarks.ops.tensorflow.graph_utils_custom import get_flops_for_node_list

    with graph.as_default():
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        profile = tf.profiler.profile(run_meta=run_meta, cmd="scope", options=opts)
        flops_all_ops = profile.total_float_ops

    graph_def = graph.as_graph_def()

    if keywords is None:
        return flops_all_ops

    keywords_flops = 0

    def _checkWordListInName(word_or_list, name):
        if isinstance(word_or_list, str):
            return word_or_list in name

        for word in word_or_list:
            if not word in name:
                return False
        return True

    for word_or_list in keywords:
        keywords_nodes = [node for node in graph_def.node if _checkWordListInName(word_or_list, node.name)]
        flops = get_flops_for_node_list(graph, keywords_nodes)
        keywords_flops += flops

    return_flops = flops_all_ops - keywords_flops if exclude_keywords else keywords_flops
    return return_flops
