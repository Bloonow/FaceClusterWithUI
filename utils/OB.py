"""
OB.py @copyright Excel Bloonow
"""
import traceback
from os import getcwd


def __print_info(**kwargs):
    info = str()
    for key in kwargs:
        info += f'\033[0;31m{key}\033[0m' + ': ' + f'\033[0;32m{kwargs[key]}\033[0m' + ' | '
    print(info[:-3])


def what(*args):
    # 栈顶是 traceback.extract_stack() 的调用信息
    # 故栈顶的下一个栈帧，是该函数 what(*args) 被调用时的信息
    frame = traceback.extract_stack()[-2]
    # FrameSummary.__slots__ = ('filename', 'lineno', 'name', 'line', 'locals')
    # 其中的 line 属性包含了函数被调用时的函数名和参数名的信息，如 what(x, y, z)
    fn, ln, call = frame.filename, frame.lineno, frame.line
    where = 'At : ' + fn[len(getcwd()) + 1:] + ' ' + '{:<3}'.format(ln) + ' : ' + call
    print(f'\033[0;34m{where}\033[0m')

    # 按最大的参数名称长度对齐
    ans = call[call.find('(') + 1: call.rfind(')')].split(',')
    max_len, arg_names = 0, []
    for an in ans:
        an = str.strip(an)
        arg_names.append(an)
        max_len = len(an) if len(an) > max_len else max_len
    p_str = '{:<' + str(max_len) + '}'

    for name, arg in zip(arg_names, args):
        name = p_str.format(name)
        type_str = str(type(arg))
        if type_str in ["<class 'str'>", "<class 'list'>", "<class 'tuple'>", "<class 'dict'>"]:
            shape = len(arg)
        elif type_str == "<class 'numpy.ndarray'>":
            shape = arg.shape
        elif type_str == "<class 'torch.Tensor'>":
            shape = arg.size()
        else:
            shape = None
        __print_info(name=name, type=type_str, shape=shape, value=arg)


def log(info_str):
    print(f'\033[0;35m{info_str}\033[0m')
