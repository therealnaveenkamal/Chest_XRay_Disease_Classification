def calcloss(positivewt, negativewt, al=1e-7):
    def weighted_loss(y_true, y_pred):
        loss = 0.0
        for i in range(len(positivewt)):
            loss += -((positivewt[i] * K.transpose(y_true)[i] * K.log(K.transpose(y_pred)[i] + al))+(negativewt[i]*(1 - K.transpose(y_true)[i])*K.log(1 - K.transpose(y_pred)[i] + al)))
        return K.mean(loss)
    return weighted_loss
{"mode":"full","isActive":false}