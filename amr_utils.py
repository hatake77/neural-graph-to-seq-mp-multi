#coding=gbk
def read_anonymized(amr_lst, amr_node, amr_edge):
    assert sum(x=='(' for x in amr_lst) == sum(x==')' for x in amr_lst) # 判断括号关系是否正确
    cur_str = amr_lst[0]  #AMR图第一个必然是节点，预先加进去
    cur_id = len(amr_node) #cur_id只有在最开始进入函数的时候被赋值
    amr_node.append(cur_str) #赋值之后就被添加进nodes数组

    i = 1
    while i < len(amr_lst):
        if amr_lst[i].startswith(':') == False: ## cur cur-num_0 不以'：'开头的是node节点，两个节点直接邻近，默认边是value
            nxt_str = amr_lst[i]
            nxt_id = len(amr_node)
            amr_node.append(nxt_str)
            amr_edge.append((cur_id, nxt_id, ':value'))
            i = i + 1
        elif amr_lst[i].startswith(':') and len(amr_lst) == 2: ## cur :edge 边后面未跟节点，节点设为unk
            nxt_str = 'num_unk'
            nxt_id = len(amr_node)
            amr_node.append(nxt_str)
            amr_edge.append((cur_id, nxt_id, amr_lst[i]))
            i = i + 1
        elif amr_lst[i].startswith(':') and amr_lst[i+1] != '(': ## cur :edge nxt 不再有下一层
            nxt_str = amr_lst[i+1]
            nxt_id = len(amr_node)
            amr_node.append(nxt_str)
            amr_edge.append((cur_id, nxt_id, amr_lst[i]))
            i = i + 2  # 节点已经被加进去，直接跳过
        elif amr_lst[i].startswith(':') and amr_lst[i+1] == '(': ## cur :edge ( ... ) 后面有下一层，递归
            number = 1
            j = i+2
            #找出括号内AMR图的范围
            while j < len(amr_lst):
                number += (amr_lst[j] == '(')
                number -= (amr_lst[j] == ')')
                if number == 0:
                    break
                j += 1
            assert number == 0 and amr_lst[j] == ')', ' '.join(amr_lst[i+2:j])
            nxt_id = read_anonymized(amr_lst[i+2:j], amr_node, amr_edge) #递归分解，括号内部又是一个AMR解析树
            amr_edge.append((cur_id, nxt_id, amr_lst[i]))  #cur_id不会变，实现describe与person是0-1的连接
            i = j + 1
        else:
            assert False, ' '.join(amr_lst)
    return cur_id

if __name__ == '__main__':
    for path in ['data/dev-dfs-linear_src.txt', 'data/test-dfs-linear_src.txt', 'data/training-dfs-linear_src.txt', ]:
        print(path)
        for i, line in enumerate(open(path, 'rU')):
            amr_node = []
            amr_edge = []
            read_anonymized(line.strip().split(), amr_node, amr_edge)
