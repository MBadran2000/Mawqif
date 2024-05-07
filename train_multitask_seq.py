from train_multitask_final import run
import config 
import nlpaug.augmenter.word as naw

if __name__ == '__main__': 
  if config.USE_NLPAUG:
    nlp_aug = [naw.RandomWordAug(action='delete', aug_p=config.NLPAUG_PROB),naw.SynonymAug(aug_src='ppdb', lang='arb',model_path ='/home/dr-nfs/m.badran/mawqif/ppdb-1.0-s-lexical',aug_p=config.NLPAUG_PROB)]
    print(nlp_aug)
  else:
    nlp_aug = None

  for i in range(16,26): 
      config.CONTRASTIVE_LOSS = i
      config.Version = "V104."+str(i)
      # run()
      try:
        print(config.CONTRASTIVE_LOSS, config.Version)
        run(nlp_aug)
        print("completed ",str(i))
      except:
        print("Something went wrong ", str(i))