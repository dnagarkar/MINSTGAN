#generate a fake data batch using the generator
fake_data_batch = G.forward(noise)


#Calculate the discriminator output for real and fake data
real_scores = D.forward(real_images)
fake_scores = D.forward(fake_data_batch)
