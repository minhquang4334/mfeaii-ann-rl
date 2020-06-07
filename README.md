# mfea_ii_ann
## Motivation
People rarely solve problems without any knowledge, knowledge about the problem and link solutions from related issues to each other. This observation is the driving force in the development of multitasking evolution algorithms through the exchange of knowledge among related problems in which each problem will be considered as a task that can be solved. time. Good solutions between tasks are exchanged for improved performance on each task. However, whether this exchange is always effective or not depends on the relationship between them. In cases where there is no or little relationship between them, the exchange of information will most likely lead to "negative exchange". Meaning that instead of speeding up each other optimally, they will result in a decrease in the speed of convergence on each task. This is also a problem encountered by the first generation multitasking evolution algorithm. So, the multitasking evolution algorithm with online exchange coefficient estimation (MFEA-II) was born to allow us to understand the relationship between tasks based directly on the data generated during the optimization process. , thereby exploiting complementarities between tasks more effectively.

Besides, neural network training problem is a problem that is very noticeable in the field of artificial intelligence. Along with training a common neural network, training multiple neural networks simultaneously to take advantage of complementary between networks is also a huge challenge. Especially the application of intensive math problems, because in this environment the application of methods related to derivatives is increasingly showing limitations. With the idea of ​​MFEA-II, I believe that the algorithm has a great potential in solving the above challenge. However, in my understanding, the application of multitasking evolution algorithm to train multiple neural networks at the same time is still a new direction, especially there have been no studies applying MFEA-II to solve. This problem.

## Motivation


## Configuration
>virtualenv venv

>source venv/bin/activate

>pip3 install -r requirement.txt

>python3 run_init.py

>python3 run_experiment.py
