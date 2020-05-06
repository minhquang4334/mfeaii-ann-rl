from .helpers import *
from .instance import *

def export_result(instances):
    ''' print template
    CEA (4,5,6) & $0.0331 \pm 0.0091 $& $0.0166 \pm 0.0097$ & $\mathbf{0.0058 \pm 0.0012}$ \\ 
    MFEA (4,5,6) & $0.0268 \pm 0.0078$ & $\mathbf{0.0116 \pm 0.0034}$ & $0.0068 \pm 0.0016$ \\ 
    MFEAII (4,5,6)  & $\mathbf{0.0260 \pm 0.01}$ & $0.0163 \pm 0.0083$ & $0.0140 \pm 0.0106$ \\ \hline
    CEA (6,7,8) & $0.0115 \pm 0.0083$ & $0.0072 \pm 0.0064$ & $0.0026 \pm 0.0018$ \\ 
    MFEA (6,7,8) & $\mathbf{0.0082 \pm 0.0051}$ & $\mathbf{0.0029 \pm 0.0012}$ & $\mathbf{0.0012 \pm 0.0009}$ \\ 
    MFEAII (6,7,8)  & $0.0091 \pm 0.0066$ & $0.0038 \pm 0.0024$ & $0.0013 \pm 0.001$ \\ \hline
    '''
    K = 3
    results = []
    texts = ''
    for instance in instances:
        result = group_result_by_index(instance, 1, K)
        results.append(result)

    for re in results:
        item_template = '& ${} \\pm {}$ '
        item_bold_template = ' & $\\mathbf {} \\pm {}$ '
        for tmp in re:
            index = 0
            text = ''
            for item in tmp:
                text += item_template.format(item[2], item[3])
                index += 1
            text += '\\\\ \n'
            texts += text
            print (text)
    return text

def result_to_string():
    list_instances = get_list_instance_name()
    results = []
    for ins in list_instances:
        instance = Instance(config, ins)
        result = instance.best_results()
        results.append(result)
        print(ins, result)
    export_result(results)

if __name__ == "__main__":
    pass