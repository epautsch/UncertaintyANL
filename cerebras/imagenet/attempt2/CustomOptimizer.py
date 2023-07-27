import torch
import cerebras_pytorch.experimental as cstorch


class SGD(cstorch.optim.Optimizer):
    def __init__(
        self,
        params,
        cls_token,
        pos_embed,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        maximize=False,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                f"Nesterov momentum requires a `momentum` and zero `dampening`. "
                f"`momentum` was {momentum} and `dampening` was {dampening}."
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )

        super().__init__(params, defaults)
        self.cls_token = cls_token
        self.pos_embed = pos_embed

    def state_names_to_sparsify(self):
        return ["momentum_buffer"]

    def preinitialize(self):
        for group in self.param_groups:
            for p in group['params']:
                if group['momentum'] != 0:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(
                        p, device="cpu"
                    ).to(p.device)

        # init momentum buffers for cls_token and pos_embed if needed
        if self.defaults['momentum'] != 0:

            self.state[self.cls_token] = {'momentum_buffer': torch.zeros_like(
                self.cls_token, device='cpu'
            ).to(self.cls_token.device)}
            
            self.state[self.pos_embed] = {'momentum_buffer': torch.zeros_like(
                self.pos_embed, device='cpu'
            ).to(self.pos_embed.device)}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group['momentum']
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                self._update_param(p, lr, weight_decay, momentum, dampening, nesterov, maximize)

            # update cls and pos
            if self.cls_token.grad is not None:
                self._update_param(self.cls_token, lr, weight_decay, momentum, dampening, nesterov, maximize)

            if self.pos_embed.grad is not None:
                self._update_param(self.pos_embed, lr, weight_decay, momentum, dampening, nesterov, maximize)

        return loss

    def _update_param(self, p, lr, weight_decay, momentum, dampening, nesterov, maximize):
        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("SGD does not support sparse gradients.")

        grad = grad if not maximize else -grad

        if weight_decay != 0:
            grad = grad.add(p, alpha=weight_decay)

        if momentum != 0:
            buf = self.state[p]["momentum_buffer"]

            buf.mul_(momentum).add_(grad, alpha=1.0 - dampening)

            if nesterov:
                grad.add_(buf, alpha=momentum)
            else:
                grad = buf

        p.add_(-lr * grad)

