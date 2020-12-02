import cv2
import os
import tensorflow as tf
import numpy as np
keras = tf.keras
layers = keras.layers
image = keras.preprocessing.image
latent_dim = 80
height = 80
width = 80
channels = 3


generator_input = keras.Input(shape=(latent_dim,))
x = layers.Dense(256 * 40 * 40)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((40, 40, 256))(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)
generator.summary()

discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.8)(x)
x = layers.Dense(1, activation='sigmoid')(x)
discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()
discriminator_optimizer = keras.optimizers.RMSprop(
    lr=0.0008,
    clipvalue=1.0,
    decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer,
                      loss='binary_crossentropy')


discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
print(gan_input)
print(generator(gan_input))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
gan.load_weights('gan.h5')

def get_trainData(start, end):
    imgs = []
    for i in range(end-start):
        img = cv2.imread(
            '.\\DogAndCat\\train\\dogs\\dog.'+str(start+i)+'.jpg')
        imgs.append(cv2.resize(img, (80, 80)))
    return np.array(imgs)/255.0


# get_trainData(100, 200)
# print(get_trainData(100, 200).shape)
# //100 * 300 * 300 * 3
train_length = 12500
iterations = 20000
batch_size = 10
save_dir = 'imgDir'
start = 0


for step in range(iterations):
    print(str(step)+'\\'+str(iterations))
    random_latent_vectors = np.random.normal(size=(batch_size,
                                                   latent_dim))
    generated_images = generator.predict(random_latent_vectors)
    stop = start + batch_size
    real_images = get_trainData(start, stop)
    combined_images = np.concatenate([generated_images, real_images])
    labels = np.concatenate([np.ones((batch_size, 1)),
                             np.zeros((batch_size, 1))])
    labels += 0.05 * np.random.random(labels.shape)
    d_loss = discriminator.train_on_batch(combined_images, labels)
    random_latent_vectors = np.random.normal(size=(batch_size,
                                                   latent_dim))
    misleading_targets = np.zeros((batch_size, 1))
    a_loss = gan.train_on_batch(random_latent_vectors,
                                misleading_targets)
    start += batch_size
    if start == 12500:
        start = 0
    if start > train_length - batch_size:
        start = 0
    # 每100次训练保存一次
    if step % 100 == 0:        
        gan.save_weights('gan.h5')
        print('discriminator loss:', d_loss)
        print('adversarial loss:', a_loss)
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir,
                              'generated_dog' + str(step) + '.png'))
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir,
                              'real_frog' + str(step) + '.png'))
