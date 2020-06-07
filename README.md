# mfea_ii_ann
## Motivation
Con người hiếm khi giải quyết vấn đề mà không có chút hiểu biết, kiến thức gì về vấn đề đó và liên kết việc giải pháp từ những vấn đề liên quan với nhau. Quan sát này chính là động lực trong việc xây dựng thuật toán tiến hóa đa nhiệm thông qua việc trao đổi tri thức giữa các bài toán có liên quan trong đó mỗi bài toán sẽ được coi như là các tác vụ có thể giải quyết được đồng thời. Các nghiệm tốt giữa các tác vụ được trao đổi với nhau để cải thiện hiệu suất tối ưu trên từng tác vụ. Tuy nhiên, việc trao đổi này thực sự có luôn đem lại hiệu quả hay không thì còn phụ thuộc vào mối quan hệ giữa chúng.
Trong trường hợp giữa chúng không có hoặc ít có mối quan hệ với nhau thì việc trao đổi thông tin rất có thể sẽ dẫn đến "trao đổi âm". Có nghĩa là thay vì tăng tốc độ tối ưu cho nhau, chúng sẽ dẫn đến việc gây ra giảm tốc độ hội tụ trên từng tác vụ. Đây cũng là một vấn đề mà thuật toán tiến hóa đa nhiệm thế hệ đầu gặp phải. Vậy nên, thuật toán tiến hóa đa nhiệm với ước lượng hệ số trao đổi trực tuyến (MFEA-II) đã ra đời cho phép hiểu được mối quan hệ giữa các tác vụ dựa trực tiếp vào dữ liệu sinh ra trong quá trình tối ưu, từ đó khai thác sự bổ trợ giữa các tác vụ một cách hiệu quả hơn. 

Bên cạnh đó, bài toán huấn luyện mạng neural là một vấn đề đang rất được chú ý trong lĩnh vực trí tuệ nhân tạo. Cùng với huấn luyện 1 mạng neural thông thường, việc huấn luyện nhiều mạng neural đồng thời để tận dụng sự bổ trợ giữa các mạng cũng là một thách thức rất lớn. Đặc biệt là việc áp dụng vào các bài toán học tăng cường, do trong môi trường này việc áp dụng các phương pháp liên quan đến đạo hàm đang ngày càng thể hiện những hạn chế.
Với ý tưởng của MFEA-II, tôi tin rằng thuật toán có tiềm năng lớn trong việc giải quyết thách thức kể trên. Mặc dù vậy, trong tầm hiểu biết của tôi, việc áp dụng giải thuật tiến hóa đa nhiệm để huấn luyện nhiều mạng neural đồng thời vẫn là một hướng đi mới, đặc biệt chưa có nghiên cứu nào áp dụng MFEA-II để giải quyết bài toán này.

## Motivation


## Configuration
>virtualenv venv

>source venv/bin/activate

>pip3 install -r requirement.txt

>python3 run_init.py

>python3 run_experiment.py
