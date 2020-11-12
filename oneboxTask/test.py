class DeepLightLinearFunction(torch.autograd.Function):

   # Note that both forward and backward are @staticmethods
   @staticmethod
   # bias is an optional argument
   def forward(ctx, input, weight, bias=None):
       ctx.save_for_backward(input, weight, bias)
       #output = input.mm(weight.t())
       output = torch.tensor(phcore.matmul(input.detach().numpy(),
                       weight.t().detach().numpy()),
                       requires_grad=True).type('torch.FloatTensor')
       if bias is not None:
           output += bias.unsqueeze(0).expand_as(output)
       return output

   # This function has only a single output, so it gets only one gradient
   @staticmethod
   def backward(ctx, grad_output):
       # This is a pattern that is very convenient - at the top of backward
       # unpack saved_tensors and initialize all gradients w.r.t. inputs to
       # None. Thanks to the fact that additional trailing Nones are
       # ignored, the return statement is simple even when the function has
       # optional inputs.
       input, weight, bias = ctx.saved_tensors
       grad_input = grad_weight = grad_bias = None

       # These needs_input_grad checks are optional and there only to
       # improve efficiency. If you want to make your code simpler, you can
       # skip them. Returning gradients for inputs that don't require it is
       # not an error.
       grad_output.type('torch.FloatTensor')
       if ctx.needs_input_grad[0]:
           #grad_input = grad_output.mm(weight)
           print("0",grad_output.detach().numpy().shape,  weight.detach().numpy().shape)
           grad_input = torch.tensor(phcore.matmul(grad_output.detach().numpy(),
                       weight.detach().numpy()),
                       requires_grad=True).type('torch.FloatTensor')
       if ctx.needs_input_grad[1]:
           #grad_weight = grad_output.t().mm(input)
           print("1",grad_output.t().detach().numpy().shape, input.detach().numpy().shape)
           grad_weight = torch.tensor(phcore.matmul(grad_output.t().detach().numpy(),
                       input.detach().numpy()),
                       requires_grad=True).type('torch.FloatTensor')

       if bias is not None and ctx.needs_input_grad[2]:
           grad_bias = grad_output.sum(0)

       return grad_input, grad_weight, grad_bias