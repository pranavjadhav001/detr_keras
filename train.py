import time
def main(epochs,datagen,loss,model,optimizer):
    for _ in range(epochs):
        batch_loss = []
        start = time.time()
        for batch, (images, labels) in enumerate(datagen):
            with tf.GradientTape() as tape:
                outputs = model(images, training=True)
                pred_loss = []
                #for output, label, loss_fn in zip(outputs, labels, loss):
                #    pred_loss.append(loss_fn(label, output))
                total_loss = loss(outputs,labels)
                batch_loss.append(total_loss)
                print('total:',total_loss)
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables))
        print('time taken:',(time.time()-start),'loss:',np.mean(batch_loss))
