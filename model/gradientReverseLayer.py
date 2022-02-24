from torch.autograd import Function


class GRL(Function):
    @staticmethod
    def forward(ctx, input, lamda):
        ctx.lamda = lamda
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lamda, None
