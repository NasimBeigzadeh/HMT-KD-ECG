class MultiTeacherKDLoss(nn.Module):

    def __init__(
            self,
            temperature=2.0,
            alpha=0.2
    ):
        super().__init__()

        self.temperature = temperature
        self.alpha = alpha

        self.ce_loss = nn.CrossEntropyLoss()


    def forward(
            self,
            student_output,
            teacher_output,
            labels
    ):

        T = self.temperature


        kd_loss = nn.KLDivLoss(
            reduction="batchmean"
        )(
            F.log_softmax(
                student_output / T,
                dim=1
            ),

            F.softmax(
                teacher_output / T,
                dim=1
            )
        )


        kd_loss = self.alpha * kd_loss


        ce_loss = (
            1 - self.alpha
        ) * self.ce_loss(
            student_output,
            labels
        )


        return kd_loss + ce_loss
