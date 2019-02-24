import dynet as dy

class Model(object):
    def __init__(self):
        self.m = dy.ParameterCollection()
        self.w = self.m.add_parameters((8,2))
        self.v = self.m.add_parameters((1,8))
        self.b = self.m.add_parameters((8))
        self.trainer = dy.SimpleSGDTrainer(self.m)

    def cal_loss(self, inputs, expected_answer):
        dy.renew_cg()
        x = dy.vecInput(2)
        x.set(inputs)
        output = dy.logistic(self.v*(dy.tanh(self.w*x+self.b)))
        y = dy.scalarInput(expected_answer)
        loss = dy.binary_log_loss(output, y)
        return loss

    def batch_train_step(self, inputs, expected_answer):
        losses = []
        for i in range(len(inputs)):
            losses.append(self.cal_loss(inputs[i], expected_answer[i]))
        ave_loss = sum(losses)/len(inputs)
        ave_loss.forward()
        ave_loss.backward()
        self.trainer.update()
        return ave_loss.value()

    def train_step(self, inputs, expected_answer):
        los = self.cal_loss(inputs, expected_answer)
        los.forward()
        los.backward()
        self.trainer.update()
        return los.value()


    def eval_step(self, inputs, expected_answer):
        dy.renew_cg()
        x = dy.vecInput(2)
        x.set(inputs)
        output = dy.logistic(self.v * (dy.tanh(self.w * x + self.b)))
        y = dy.scalarInput(expected_answer)
        loss = dy.binary_log_loss(output, y)
        return loss.value(), output.value()

def create_xor_instances(num_rounds=2000):
    questions = []
    answers = []
    for round in range(num_rounds):
        for x1 in range(2):
            for x2 in range(2):
                answer = 0 if x1 == x2 else 1
                questions.append((x1, x2))
                answers.append(answer)
    return questions, answers

def create_xor_instance_batches(num_rounds=2000):
    questions = []
    answers = []
    for round in range(num_rounds):
        for x1 in range(2):
            for x2 in range(2):
                answer = 0 if x1 == x2 else 1
                questions.append(((x1, x2), (x1, x2)))
                answers.append((answer, answer))
    return questions, answers

questions, answers = create_xor_instance_batches()
total_loss = 0
seen_instances = 0

model = Model()

for question, answer in zip(questions, answers):
    print(question, answer, len(question))
    seen_instances += 1
    total_loss += model.batch_train_step(question, answer)

    if seen_instances > 1 and seen_instances%100 == 0:
        print("average loss is:", total_loss/seen_instances)
        for i in range(2):
            for j in range(2):
                _, value = model.eval_step((i,j), 0 if i == j else 1)
                print(i, j, value)
