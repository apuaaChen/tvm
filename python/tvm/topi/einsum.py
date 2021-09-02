# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,consider-using-enumerate,redefined-outer-name
"""Einsum operator"""
from tvm import auto_scheduler
import re
from tvm import te
import tvm
def parse_operand_subscripts(subscript, shape, op_label_dict):
    """Parse the subscripts for one operand into an output of 'ndim' labels

    Parameters
    ----------
    subscript : string
        the subscript of the operand to be parsed.
    shape : tuple of int
        the shape of the input operand
    op_label_dict : dict
        a dict with pair (label<str>, dim_size<int>)
    Returns
    -------
    op_labels : list of str/int
        The parsing results
        For Example:
            subscripts="abbcbc",  ndim=6 -> op_labels=['a', 'b', 'b', 'c', 'b', 'c'].
            subscripts="ab...bc", ndim=6 -> op_labels=['a', 'b', '1', '0', 'b', 'c'].
        dimensions to be broadcast are labeled with number in decreasing order
    """
    op_labels = []
    ndim = len(shape)

    # Step 1: split the subscript by the ellipsis "@"
    subscript_break = subscript.split('@')
    assert len(subscript_break) <= 2, "einstein sum subscript string %s contains more than one ellipsis" % subscript

    sdim = 0
    for s in subscript_break:
        sdim += len(s)

    num_implicit = ndim - sdim
    assert num_implicit >= 0, "einstein sum subscript string %s doesn't match the input shape."

    # Step 2: parse the subscript
    for label in subscript_break[0]:
        if label.isalpha():
            op_labels.append(label)
        else:
            raise ValueError("'%s' is not valid for subscript %s" % (label, subscript))
    
    if len(subscript_break) == 2:
        # the broadcast indices are labeled in decreasing order
        # as it requires matching of shape from right to left
        for n in range(num_implicit):
            op_labels.append('^%d' % (num_implicit - n - 1))
        
        for label in subscript_break[1]:
            if label.isalpha():
                op_labels.append(label)
            else:
                raise ValueError("'%s' is not valid for subscript %s" % (label, subscript))

    # Step 3: bind the shape to the labels    
    assert len(op_labels) == ndim

    for idx, label in enumerate(op_labels):
        if label in op_label_dict:
            if op_label_dict[label] != shape[idx]:
                if op_label_dict[label] == 1:
                    op_label_dict[label] = shape[idx]
                else:
                    assert shape[idx] == 1, "the dimenions labeled with the same subscript %s cannot be broadcasted" % label
            # assert op_label_dict[label] == shape[idx], "the dimenions labeled with the same subscript %s doesn't match in size" % label
        else:
            op_label_dict[label] = shape[idx]
    
    return op_labels


def parse_output_subscripts(subscript, broadcast_labels):
    """Parse the subscripts fro the output into an output that includes 'ndim_broadcast'
       unlabeld dimensions
    
    Parameters
    ----------
    subscripts : string
        the subscripts for output to be parsed
    broadcast_label : list of strings
        the broadcast dimension labels
    
    Returns
    -------
    out_labels : list of str
        similar to the op_labels in parse_operand_subscripts, for each
        dimension, the character of the corresponding label
        zero for the broadcasting dim
    """
    out_labels = []
    ellipsis = -1
    for label in subscript:
        # A proper for an axis
        if label.isalpha():
            assert label not in out_labels, "einstein sum subscripts string includes output subscript '%s' multiple times" % label
            out_labels.append(label)
        elif label == '@':
            assert ellipsis == -1, "einstein sum subscripts string contains too many ellipsis"
            ellipsis = 1
            out_labels += broadcast_labels
        else:
            raise ValueError("'%s' is not valid for subscript" % label)
    
    if '@' not in subscript:
        assert len(broadcast_labels) == 0, "output has more dimensions than subscripts given in einsten sum, but no ellipsis provided to broadcast the extra dimensions."
    
    return out_labels


def einsum(subscripts, *operand, auto_scheduler_rewritten_layout=""):
    """Evaluates the Einstein summation convention on the operands.

    Parameters
    ----------
    subscripts : string
        Specifies the subscripts for summation as comma separated list of subscript labels.
        An implicit (classical Einstein summation) calculation is performed unless the
        explicit indicator ‘->’ is included as well as subscript labels of the precise
        output form.

    a_tuple : tuple of tvm.te.Tensor
        These are the Tensors for the operation.
        The only difference of einsum between in tvm and numpy is it needs an extra brackets
        for the tensors. For example, topi.einsum("ij, jk -> ik", (A, B)).
    
    auto_scheduler_rewritten_layout: Optional[str] = ""
        The layout after auto-scheduler's layout rewrite pass.

    Returns
    -------
    out : tvm.te.Tensor
        The calculation based on the Einstein summation convention.
    """
    # Instead of using the TIR module in tvm/include/tvm/topi/einsum.h
    # We use a more general version of einsum defined in python
    # This potentially expand the eligible search space
    nop = len(operand)
    # strip all the whitespaces for simplicity
    subscripts = subscripts.replace(" ", "")
    # replace the elipsis "..." with "@"
    subscripts = subscripts.replace("...", "@")
    labels = re.split(',|->', subscripts)
    if '->' in subscripts:
        assert len(labels) == nop + 1, "#operand doesn't match the subscripts. Expected %d, got %d" % (len(labels) -1, nop)
    else:
        assert len(labels) == nop, "#operand doesn't match the subscripts. Expected %d, got %d" % (len(labels), nop)

    # Step 1: Parse the subscripts string
    op_labels_dict = {}
    op_labels = []

    # Step 2: Parse the subscripts string
    for iop in range(nop):
        op_labels.append(parse_operand_subscripts(labels[iop], operand[iop].shape, op_labels_dict))

    # get the broadcast labels
    broadcast_labels = []
    for op_label in op_labels:
        for label in op_label:
            if '^' in label and label not in broadcast_labels:
                broadcast_labels.append(label)

    broadcast_labels = sorted(broadcast_labels, reverse=True)

    # Step 3: Get the output labels
    if '->' not in subscripts:
        # If there is no output signature, fill output_labels and ndim_output
        # using each label that appeared once, in alphabetical order
        # If no output was specified, always broadcast left
        # the remaining labels are sorted in alphabetical order
        # get the operator labels in the output & reduce labels
        unsorted_labels = []
        reduce_labels = []
        for iop, op_label in enumerate(op_labels):
            for ilabel, label in enumerate(op_label):
                if '^' not in label:
                    single = True
                    for jop, op_label_j in enumerate(op_labels):
                        for jlabel, label_j in enumerate(op_label_j):
                            if (jop != iop or ilabel != jlabel) and label == label_j:
                                single = False
                                break
                    if single and label not in unsorted_labels:
                        unsorted_labels.append(label)
                    if not single and label not in reduce_labels:
                        reduce_labels.append(label)
        out_labels = broadcast_labels + sorted(unsorted_labels)
    else:
        # If the output signature is provided
        out_labels = parse_output_subscripts(labels[-1], broadcast_labels)
        reduce_labels = []
        for iop, op_label in enumerate(op_labels):
            for ilabel, label in enumerate(op_label):
                if label not in out_labels and label not in reduce_labels:
                    reduce_labels.append(label)

    # Step 4: get the output shape
    out_shape = ()
    for l in out_labels:
        out_shape += (op_labels_dict[l],)

    def einsum_compute(args):
        dim_var_dict = {}
        rd_axis = []
        for idx, l in enumerate(out_labels):
            dim_var_dict[l] = args[idx]
        for key in reduce_labels:
            dim_var_dict[key] = te.reduce_axis((0, op_labels_dict[key]), key)
            rd_axis.append(dim_var_dict[key])

        te_operands = []
        for iop in range(nop):
            index = ()
            for idx, l in enumerate(op_labels[iop]):
                # handle the broadcast situation
                if operand[iop].shape[idx] == 1:
                    index += (0,)
                else:
                    index += (dim_var_dict[l],)
            te_operands.append(operand[iop][index])
        sum_op = te_operands[0]
        for iop in range(nop):
            if iop != 0:
                sum_op *= te_operands[iop]
        if len(reduce_labels) != 0:
            return te.sum(sum_op, axis=rd_axis)
        else:
            return sum_op
    
    # Step 5: get the compute
    mat = te.compute(
        out_shape,
        lambda *dim_var: einsum_compute(dim_var),
        name="T_einsum"
    )
    
    # mat = cpp.einsum(subscripts, operand)
    if auto_scheduler_rewritten_layout:
        mat = auto_scheduler.rewrite_compute_body(mat, auto_scheduler_rewritten_layout)
    return mat
