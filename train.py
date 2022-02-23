import time
import numpy as np
import tensorflow as tf

def main(epochs,datagen,Matcher,loss,model,optimizer):
    for _ in range(epochs):
        batch_loss = []
        for batch, (images, y_true) in enumerate(datagen):
            with tf.GradientTape() as tape:
                outputs = model(images, training=True)
                pred_access_indexes = []
                y_target_logits = []
                y_target_box = []
                batch_size = len(y_true)
                batch_query_indices = Matcher(y_true,outputs[0],outputs[1]) 
                for i in range(batch_size):
                    pred_access_indexes.extend([[i,j] for j in batch_query_indices[i][0]])
                    y_target_logits.extend(y_true[i][batch_query_indices[i][1],-1])
                    y_target_box.extend(y_true[i][batch_query_indices[i][1],:4])
                y_target_box = np.array(y_target_box).reshape(-1,4)
                y_target_logits = np.array(y_target_logits,dtype=int).flatten()
                y_pred_box = tf.gather_nd(outputs[1],pred_access_indexes)
                y_pred_logits = tf.gather_nd(outputs[0],pred_access_indexes)
                total_loss = loss(y_target_logits,y_target_box,y_pred_logits,y_pred_box)
                batch_loss.append(total_loss)
                print('total:',total_loss)
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables))
        print('time taken:',(time.time()-start),'loss:',np.mean(batch_loss))