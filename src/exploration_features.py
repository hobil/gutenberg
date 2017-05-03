import pickle
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib
from collections import Counter
from metadata import load_metadata, super_categories, pos_mapping

def available_docs():
  bookids=os.listdir(FEATVEC_FOLDER) # not needed later when we have all featvecs
  bookids=[int(b.split('.')[0]) for b in bookids if 'featvecs' not in b]
  #index=filter_meta(df_meta).index
  index=df_meta.index
  bookids_text=[b for b in bookids if b in index]
  return bookids_text

def explore(df_meta,feature_series,split_param, split_param_vals = None, x_min = None, x_max = None, bin_size = 0.01, bbox_to_anchor=(1.55, 1.03),markerscale = 3, linewidth = 2.5,colors=None, marker = '.',min_y = None, mean = False):
  fig, ax = plt.subplots()
  if x_min is None:
    x_min = feature_series.min().round(2)
  if x_max is None:
    x_max = feature_series.max()
  bins = np.arange(x_min,x_max+bin_size/2,bin_size) 
  
  #xs = np.arange(x_min + bin_size/2, x_max - bin_size / 2, bin_size)
  if split_param_vals is None:
    split_param_vals = set(df_meta[split_param])
  
  y_max = - np.inf
  y_min = np.inf
  lines = []
  for n,par in enumerate(split_param_vals):
    if par is None:
      print("continuing")
      continue
    
    mask = df_meta[df_meta[split_param] == par].index
    f = feature_series.ix[mask]
    f = f[~np.isnan(f)]
  
    vals, edges = np.histogram(f, bins)
    left,right = edges[:-1],edges[1:]
    #X = np.array([left,right]).T.flatten()
    X = np.array([left,right]).mean(0)
    #Y = np.array([vals,vals]).T.flatten()
    Y = vals# / sum(vals)
    
    Y = Y / sum(vals)
    
    y_max = max(y_max,Y.max())
    y_min = min(y_min,Y.min())
    #vals = np.histogram(feature_series[mask],bins = bins)[0]
    #print(vals.shape)
    #print(xs.shape)
    #print(bins.shape)
    #plt.plot(xs,vals)
    if colors is not None:
      line = plt.plot(X,Y, linewidth = linewidth, label = str(par),color=colors[n], marker = marker)
    else:
      line = plt.plot(X,Y, linewidth = linewidth, label = str(par), marker = marker)
      #if mean:
        #ax.axvline(X.mean(), linestyle='dashed', linewidth=2, label = 'mean')
    lines+=line
  ax.set_xlim(x_min + bin_size/2, x_max - bin_size/2)
  if min_y is not None:
    y_min = min_y
  ax.set_ylim(y_min,y_max*1.02)
  #print(lines)
  #plt.legend(handles = lines, bbox_to_anchor= bbox_to_anchor, markerscale = markerscale)
  return fig, ax, lines

#fig, ax, lines = #explore(df_meta,df_meta.authoryearofbirth,split_param='lcc_class_simple',bin_size = 100)    
#ax.legend(handles = lines, bbox_to_anchor= (1.8,1), markerscale = 3)

LABELS=['avg word length',
        'stopwords share',
        #'number of sentences',
        'avg sent length in words',
        'ends with .',
        'ends with ?',
        'ends with !',
        'begins with quotation marks',
        'ends with quotations marks',
        'ADJ','ADP','ADV','CONJ','DET','NOUN','NUM','PART','PRON','PROPN','PUNCT','VERB',
        'MD','VB','VBD','VBG','VBN','VBP','VBZ',
        'TO','RP','POS',
        'NN','NNS','WP',
        'RB','WRB','WBR','EX','RBS','RBR',
        'JJ','PRP$','WDT','JJR','JJS','PDT','WP$',
        ',','.',':','HYPH','QUOT','BRACKET']


EXPLORATION_FOLDER='exploration'
plt.rcParams['figure.titlesize']='xx-large'  # fontsize of the axes title
plt.rcParams['axes.labelsize']='xx-large' # fontsize of the x any y labels
plt.rcParams['xtick.labelsize']='xx-large'
plt.rcParams['ytick.labelsize']='xx-large'
plt.rcParams['axes.ymargin']=1
plt.rcParams['axes.grid']=False
plt.rcParams['legend.fontsize']='xx-large'
COLOR="#1f7bb4"
COLOR2="g"
COLOR3="#dd0000"
QUANTILE_SMALL = 0.1
QUANTILE_BIG = 0.9
QUANTILE_LINESTYLE = ':'
QUANTILE_LINEWIDTH = 4
MEAN_LINESTYLE = 'dashed'
MEAN_LINEWIDTH = 4

NUMBER_OF_DOCS='Number of Documents'

df_meta = load_metadata()
FEATVEC_FOLDER='../res/featvec'
bookids=sorted(available_docs())
fvs=np.array([pickle.load(open(os.path.join(FEATVEC_FOLDER,'%d.fv.pickle'%i),'rb')) for i in bookids])
fvs_all=np.array([pickle.load(open(os.path.join(FEATVEC_FOLDER,f),'rb')) for f in os.listdir(FEATVEC_FOLDER) if f[0] != 'f'])
df=pd.DataFrame(fvs,index=bookids,columns=LABELS)



def add_mean_and_quantiles(series, axes = plt,legend = True,quantile_small = QUANTILE_SMALL, quantile_big = QUANTILE_BIG,quantile_linewidth = QUANTILE_LINEWIDTH,quantile_linestyle = QUANTILE_LINESTYLE, mean_linewidth = MEAN_LINEWIDTH, mean_linestyle = MEAN_LINESTYLE ):
  axes.axvline(df.ix[:,k].quantile(quantile_small), color='k', linestyle=quantile_linestyle, linewidth=quantile_linewidth, label = '%d %% quantile'%(int(100*quantile_small)))
  axes.axvline(z.mean(), color='k', linestyle=mean_linestyle, linewidth=mean_linewidth, label = 'mean')
  axes.axvline(df.ix[:,k].quantile(quantile_big), color='k', linestyle=quantile_linestyle, linewidth= quantile_linewidth, label = '%d %% quantile'%(int(100*quantile_big)))
  if legend:
    plt.legend()
  
  
"""
######   SIZES    ####
d_size=pickle.load(open('../res/sizes.pickle','rb'))
sizes_bytes=np.array([v for k,v in d_size.items() if int(k.split('-')[0].split('/')[0].split('.')[0]) in df_meta.index])
sizes_mb=sizes_bytes/1024/1024

## ALL SIZES ###

BINS=np.arange(0,16,0.500)

fig = plt.figure()
axes=plt.gca()
maxhist=max(np.histogram(sizes_mb,bins=BINS)[0])
axes.set_ylim([0,50000])
#plt.xscale('symlog')
plt.yscale('symlog')
plt.hist(sizes_mb,bins=BINS,color=COLOR)
#fig.suptitle('test title')
plt.xlabel('Size in MiB')
plt.ylabel('Number of Documents (Logarithmic)')
#plt.axvline(sizes_mb.mean(), color='b', linestyle='dashed', linewidth=2)
i=5
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_docs_per_size.pdf'%i),bbox_inches='tight')



## SIZES UNDER 1 MB ##
small_docs=sizes_mb[sizes_mb<1]*1024
BINS=range(0,1026,25)

fig = plt.figure()
axes=plt.gca()
maxhist=max(np.histogram(small_docs,bins=BINS)[0])
axes.set_ylim([0,maxhist*1.05])
axes.set_xlim([0,1025])
plt.hist(small_docs,bins=BINS,color=COLOR)
plt.xlabel('Size in kiB')
plt.ylabel('Number of Documents')
i=6
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_docs_per_size_under_MB.pdf'%i),bbox_inches='tight')
"""


######    AVG WORD LENGTH       ######
k=0
z=fvs[:,k]
BINS=np.arange(3.0,6.0,0.5/10)

fig = plt.figure()
axes=plt.gca()
axes.set_ylim([0,3500])
axes.set_xlim([3.0,6.0])
#plt.yscale('symlog')
plt.hist(z,bins=BINS,color=COLOR)
#fig.suptitle('test title')
plt.xlabel('Average word length in the document')
plt.ylabel('Number of Documents')
add_mean_and_quantiles(df.ix[:,k])
i=7
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_docs_per_word_length.pdf'%i),bbox_inches='tight')


## AVG WORD LENGTH CATEGORY

fig, ax, lines = explore(df_meta,df['avg word length'],'lcc_class_simple',bin_size = 0.1, x_min = 3.4, marker = '.', linewidth = 2.5)
ax.legend(handles = lines, bbox_to_anchor= (1.8,1.022), markerscale = 3)
#ax.legend(handles = lines, bbox_to_anchor= (0.92,1.45), markerscale = 3)
plt.xlabel('Average word length per document')
plt.ylabel('Relative frequency')
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_word_length_per_category.pdf'%i),bbox_inches='tight')

## AVG WORD LENGTH AUTHOR
fig, ax, lines = explore(df_meta,df['avg word length'],'author',['Twain, Mark', 'Lytton, Edward Bulwer Lytton, Baron', 'Shakespeare, William'],bin_size = 0.1, x_min = 3.4, marker = '.', linewidth = 2.5)
ax.legend(handles = lines, bbox_to_anchor= (1.5,1.022), markerscale = 3)
#ax.legend(handles = lines, bbox_to_anchor= (0.95,1.25), markerscale = 3)
plt.xlabel('Average word length per document')
plt.ylabel('Relative frequency')
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_word_length_per_author.pdf'%i),bbox_inches='tight')





twain = df.ix[df_meta.author == 'Twain, Mark','avg word length']
lytton = df.ix[df_meta.author == 'Lytton, Edward Bulwer Lytton, Baron','avg word length']
shakespear = df.ix[df_meta.author == 'Shakespeare, William','avg word length']


fig, ax = plt.subplots()
common_params = dict(bins=np.arange(3.5,5.01,0.05), 
                     range=(3.5, 5), 
                     normed=True)
ax.set_xlim([3.5,5])
ax.set_ylim([0,10])
common_params['histtype'] = 'step'
plt.title('With steps')
plt.hist(twain, **common_params)
plt.hist(lytton, **common_params)
plt.hist(shakespear, **common_params)
plt.xlabel('Average word length per document')
plt.ylabel('Document density')



"""
fig, ax = plt.subplots()
BINS = np.arange(3.5,5.0,0.05)
plt.hist(twain,bins = BINS,color= COLOR,fill = None)
plt.hist(lytton,bins = BINS,color= COLOR2,fill = None)
plt.hist(shakespear,bins = BINS,color= COLOR3,fill = None)
ax.set_x_label('Average word length per document')
ax.set_y_label('Document density')

"""


fig, ax, lines = explore(df_meta,df['avg word length'],'author',['Twain, Mark','Shakespeare, William','Lytton, Edward Bulwer Lytton, Baron'],colors = [COLOR,COLOR2,COLOR3],bin_size = 0.05, marker = '')

#explore(df_meta,df_meta.authoryearofbirth,split_param='lcc_class_simple',bin_size = 100)    
ax.legend(handles = lines, bbox_to_anchor= (0.94,1.25), markerscale = 3)
plt.xlabel('Average word length per document')
plt.ylabel('Document density')



fig, ax, lines = explore(df_meta,df['avg word length'],'lcc_class',['PS','PZ','J','R','G'],bin_size = 0.1)
ax.legend(handles = lines, bbox_to_anchor= (0.94,1.25), markerscale = 3)
plt.xlabel('Average word length per document')
plt.ylabel('Document density')


"""
lccs=set(df_meta.lcc_class)
lccs_mean=[]
for l in lccs:
  lccs_mean+=[(l,df.ix[df_meta[df_meta.lcc_class.apply(lambda x: x==l)].index,k].mean())]
l1=sorted(lccs_mean,key=lambda x: x[1])
"""

# only lcc parent class
lccs=set.union(*[s for s in df_meta.named_LCC_1])
lccs_mean=[]
for l in lccs:
  lccs_mean+=[('%.3f'%df.ix[df_meta[df_meta.named_LCC_1.apply(lambda x: x!=None and l in x)].index,k].mean(),l)]
l1=sorted(lccs_mean,key=lambda x: x[0])

print("shortest words:")
for x in l1[:3]:
  print(x)

print("\nlongest words:")
for x in l1[::-1][:3]:
  print(x)

# whole lcc tag
lccs=set.union(*[s for s in df_meta.named_LCC])
lccs_mean=[]
for l in lccs:
  lccs_mean+=[(l,df.ix[df_meta[df_meta.named_LCC.apply(lambda x: x!=None and l in x)].index,k].mean())]
l2=sorted(lccs_mean,key=lambda x: x[1])


"""
lccs=set.union(*[s for s in df_meta.named_LCC_1])
l=[s for s in lccs if s[0]=='J'][0]
ls=df.ix[df_meta[df_meta.named_LCC_1.apply(lambda x: x!=None and l in x)].index,k]

BINS=30
fig = plt.figure()
axes=plt.gca()
maxhist=max(np.histogram(ls,bins=BINS)[0])
axes.set_ylim([0,maxhist*1.05])
axes.set_xlim([3.0,5.5])
#plt.yscale('symlog')
plt.hist(ls,bins=BINS)
"""


####### AVG WORDS IN SENT ######

k=2
z=df.ix[:,k]
#z2=np.sort(z)[:-85]
z2=z[z<70]
BINS=np.arange(0,70,1)

fig = plt.figure()

axes=plt.gca()
maxhist=max(np.histogram(z2,bins=BINS)[0])
axes.set_ylim([0,maxhist*1.05])
plt.hist(z2,bins=BINS,color=COLOR)
#fig.suptitle('test title')
plt.xlabel('Average number of words per sentence in the document')
plt.ylabel('Number of Documents')
add_mean_and_quantiles(df.ix[:,k])
i=8
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_avg_words_per_sent.pdf'%i),bbox_inches='tight')



# only lcc parent class
lccs=set.union(*[s for s in df_meta.named_LCC_1])
lccs_mean=[]
for l in lccs:
  lccs_mean+=[('%.3f'%df.ix[df_meta[df_meta.named_LCC_1.apply(lambda x: x!=None and l in x)].index,k].mean(),l)]
l1=sorted(lccs_mean,key=lambda x: x[0])

print("\nleast number of words in sentence")
for x in l1[:10]:
  print(x)
print("\nmost number of words in sentence")
for x in l1[::-1][:10]:
  print(x)
print()


## AVG SENT LENGTH CATEGORY

series = df.ix[df['avg sent length in words'] < 70,'avg sent length in words']
a_series = df.ix[df_meta.lcc_class2.apply(lambda x: x is not None and x[0] == 'A'),'avg sent length in words']
authors = ['Twain, Mark', 'Lytton, Edward Bulwer Lytton, Baron', 'Shakespeare, William']

idx = df_meta[df_meta.lcc_class_simple == 'General Works'].lcc_class

fig, ax, lines = explore(df_meta,series,'lcc_class_simple',bin_size = 2, x_min = -1, x_max = 51, marker = '.', linewidth = 2.5)
ax.legend(handles = lines, bbox_to_anchor= (1.8,1.022), markerscale = 3)
#ax.legend(handles = lines, bbox_to_anchor= (0.92,1.45), markerscale = 3)
plt.xlabel('Average word length per document')
plt.ylabel('Relative frequency')
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_sent_length_per_category.pdf'%i),bbox_inches='tight')

## AVG SENT LENGTH AUTHOR
fig, ax, lines = explore(df_meta,series,'author',authors,bin_size = 1.5, x_min = -1, x_max  =41, marker = '.', linewidth = 2.5)
ax.legend(handles = lines, bbox_to_anchor= (1.5,1.022), markerscale = 3)
#ax.legend(handles = lines, bbox_to_anchor= (0.95,1.25), markerscale = 3)
plt.xlabel('Average word length per document')
plt.ylabel('Relative frequency')
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_sent_length_per_author.pdf'%i),bbox_inches='tight')








#######  STOPWORDS  ######

k=1
z=df.ix[:,k]
BINS=np.arange(0.15,0.65,0.01)

fig = plt.figure()

axes=plt.gca()
maxhist=max(np.histogram(z,bins=BINS)[0])
axes.set_ylim([0,maxhist*1.05])
axes.set_xlim([0.15,0.65])
plt.hist(z,bins=BINS,color=COLOR)
#plt.step(sorted(z),range(len(z)))
plt.xlabel('Share of stop words in the document')
plt.ylabel('Number of Documents')
add_mean_and_quantiles(df.ix[:,k])
plt.legend(loc='upper left')
i=9
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_stopwords.pdf'%i),bbox_inches='tight')


df['author']=df_meta['author']
x = df.groupby('author').mean()
plt.hist(x['stopwords share'].values,bins=BINS)


series = df['stopwords share']
authors = ['Twain, Mark', 'Lytton, Edward Bulwer Lytton, Baron', 'Shakespeare, William']

fig, ax, lines = explore(df_meta,series,'author',authors,bin_size = 0.02, x_min = 0.3, x_max  =0.7, marker = '.', linewidth = 2.5)
ax.legend(handles = lines, bbox_to_anchor= (1.7,1.022), markerscale = 3)
#ax.legend(handles = lines, bbox_to_anchor= (0.95,1.25), markerscale = 3)
plt.xlabel('Stop words share')
plt.ylabel('Relative frequency')
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_stopwords_author.pdf'%i),bbox_inches='tight')


"""
axes=plt.gca()
maxhist=max(np.histogram(z,bins=BINS)[0])
axes.set_ylim([0,len(z)])
axes.set_xlim([0,1])
plt.step(sorted(z),range(len(z)),color=COLOR)
plt.xlabel('Share of stop words in the document')
plt.ylabel('Number of Documents')
i=9
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_stopwords.pdf'%i),bbox_inches='tight')
"""

# filtering based on stopwords
k=1
z_all=fvs_all[:,k]
x=z_all[z_all<0.35]
BINS=np.arange(0,0.36,0.01)

fig = plt.figure()

axes=plt.gca()
maxhist=max(np.histogram(x,bins=BINS)[0])
axes.set_ylim([0,100])
axes.set_xlim([0,0.35])
plt.hist(z_all,bins=BINS,color=COLOR)
plt.xlabel('Share of stop words in the document')
plt.ylabel('Number of Documents')
i=10
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_stopwords_filter_out.pdf'%i),bbox_inches='tight')

df['author']=df_meta['author']
x = df.groupby('author')
plt.hist(x['stopwords share'].mean().values,bins=BINS)



## AVG SENT LENGTH CATEGORY

series = df['stopwords share']
a_series = df.ix[df_meta.lcc_class2.apply(lambda x: x is not None and x[0] == 'A'),'avg sent length in words']
authors = ['Twain, Mark', 'Lytton, Edward Bulwer Lytton, Baron', 'Shakespeare, William']

idx = df_meta[df_meta.lcc_class_simple == 'General Works'].lcc_class

fig, ax, lines = explore(df_meta,series,'lcc_class_simple',bin_size = 0.02, x_min = 0.1, x_max = 0.7, marker = '.', linewidth = 2.5)
ax.legend(handles = lines, bbox_to_anchor= (1.8,1.022), markerscale = 3)
#ax.legend(handles = lines, bbox_to_anchor= (0.92,1.45), markerscale = 3)
plt.xlabel('Average word length per document')
plt.ylabel('Relative frequency')
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_sent_length_per_category.pdf'%i),bbox_inches='tight')

## AVG SENT LENGTH AUTHOR
fig, ax, lines = explore(df_meta,series,'author',authors,bin_size = 1.5, x_min = -1, x_max  =41, marker = '.', linewidth = 2.5)
ax.legend(handles = lines, bbox_to_anchor= (1.5,1.022), markerscale = 3)
#ax.legend(handles = lines, bbox_to_anchor= (0.95,1.25), markerscale = 3)
plt.xlabel('Average word length per document')
plt.ylabel('Relative frequency')
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_sent_length_per_author.pdf'%i),bbox_inches='tight')









#######  POS  ######

n_bins = 10
x = np.random.randn(1000, 3)
fig, ax0 = plt.subplots(nrows=1, ncols=1)

colors = ['red', 'tan', 'lime']
ax0.hist(x, n_bins, normed=1, histtype='bar', color=colors, label=colors)
ax0.legend(prop={'size': 10})
ax0.set_title('bars with legend')
              
index=range(len(df))         
fig, ax = plt.subplots()
axes=plt.gca()
axes.set_ylim([0,len(df)])
axes.set_xlim([0,0.30])
colors=[v for k,v in matplotlib.colors.cnames.items()][:12]
for k,col in zip(list(range(8,20)),colors):
  #measures=np.array([sorted(df.ix[:,k]) for k in range(8,12)]).T
  ax.plot(np.array([sorted(df.ix[:,k])]).T, index, col, label=pos_mapping[df.columns[k]])
legend = ax.legend(shadow=True)

#colors = ['red', 'tan', 'lime']
#ax0.hist(measures,range(len(df)),color=colors,label=colors)
#plt.legend([1,2,3,4],['a','b','c','d'])
plt.xlabel('Share of stop words in the document')
plt.ylabel('Number of Documents')
i=11
plt.show()
#fig.savefig(os.path.join(EXPLORATION_FOLDER,'pokus.pdf'))

# PROPN 
k=16
k+=1
z=df.ix[:,k]
z=fvs_all[:,k]
BINS=np.arange(0,max(z)+0.01,0.01)

fig = plt.figure()

axes=plt.gca()
maxhist=max(np.histogram(z,bins=BINS)[0])
axes.set_ylim([0,maxhist*1.05])
#axes.set_xlim([0,max(z)+0.01])
axes.set_xlim([0,0.4])
plt.hist(z,bins=BINS,color=COLOR)
#plt.step(sorted(z),range(len(z)))
plt.xlabel(df.columns[k])
plt.ylabel('Number of Documents')
i=9
#fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_stopwords.pdf'%i),bbox_inches='tight')

#### SUBPLOTS
def generate_plot(ids,BINS,xlim,size):
  for n,k in enumerate(ids[size]):
    ax[n,size].hist(df.ix[:,k],bins=BINS[size],color=COLOR)
    ax[n,size].set_ylim([0,7000])
    ax[n,size].set_xlim(xlim[size])
    ax[n,size].set_xlabel(pos_mapping[df.columns[k]].upper())
    ax[n,size].set_ylabel('No. of documents')
    q_s=ax[n,size].axes.axvline(df.ix[:,k].quantile(QUANTILE_SMALL), color='k', linestyle=QUANTILE_LINESTYLE, linewidth=QUANTILE_LINEWIDTH, label = '%d %% quantile'%(int(100*QUANTILE_SMALL)))
    q_b=ax[n,size].axes.axvline(df.ix[:,k].quantile(QUANTILE_BIG), color='k', linestyle=QUANTILE_LINESTYLE, linewidth=QUANTILE_LINEWIDTH, label = '%d %% quantile'%(int(100*QUANTILE_BIG)))
    mean=ax[n,size].axes.axvline(df.ix[:,k].mean(), color='k', linestyle=MEAN_LINESTYLE, linewidth=MEAN_LINEWIDTH, label = 'mean')
    #add_mean_and_quantiles(df.ix[:,k],axes = ax[n,size])
    ax[n,size].legend([q_s,mean,q_b],(q_s.get_label(),mean.get_label(),q_b.get_label()))

## PLOTTING PAGE OF POS ##
#plt.rcParams['axes.labelsize']='xx-large'
SMALL=1
BIG=0
ids={BIG:[9,12,13,17,18,19],SMALL:[8,10,11,14,15,16]}
BINS={BIG:np.arange(0,0.261,0.01/2),SMALL:np.arange(0,0.131,0.01/7)}
xlim={SMALL:[0,0.13],BIG:[0,0.26]}

index=range(len(df))         
fig, ax = plt.subplots(6,2,figsize=(21,30))
generate_plot(ids,BINS,xlim,SMALL)
generate_plot(ids,BINS,xlim,BIG)

i=11
plt.show()
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_pos.pdf'%i),bbox_inches='tight')

### propn > 0.1
n1=len(df)
lcc_list=[]
for l in df_meta.ix[:,'named_LCC_1']:
    lcc_list+=list(l)
lcc_counter=Counter(lcc_list)
for x in lcc_counter.most_common(10):
  print("%.2f %%, %d, %s"%(100*x[1]/n1,x[1],x[0]))
  
  
high_propn=df[df.ix[:,17]>=0.1].index
n2=len(high_propn)
lcc_list=[]
for l in df_meta.ix[high_propn,'named_LCC_1']:
    lcc_list+=list(l)
lcc_counter2=Counter(lcc_list)
for x in lcc_counter2.most_common(10):
  print("%.2f %%, %d, %s"%(100*x[1]/n2,x[1],x[0]))

q=[]
for x in lcc_counter2:
  whole_freq=lcc_counter[x]/n1
  new_freq=lcc_counter2[x]/n2
  q+=[(new_freq/whole_freq,x)]
q=sorted(q)
for t in [1,2,3,4,5,-5,-4,-3,-2,-1]:
  x= q[t]
  print("%.2f, %s"%(x[0],x[1]))
  
  
  
###############################
# PLOTS FOR PREDICTIVE MODEL AUTHOR
n_features = [0.01,0.02,0.05, 0.1, 0.5, 1, 2, 5, 8, 10, 15, 16.790]
_bin = [0.01,0.011,0.02, 0.045, 0.259,0.488,0.635,0.722, 0.741, 0.735, 0.720, 0.707]
rel =[0.124,0.122,0.518, 0.596, 0.584,0.488,0.379,0.263,0.244,0.267,.28,0.302]
tfidf = [0.117,0.122,0.456, 0.533, 0.569,0.482,0.368,0.196, 0.205, 0.231, .25, 0.287]
text_feats = np.repeat(0.576, len(tfidf) + 2)
naiv = np.repeat(0.01, len(tfidf) + 2)
n0_features = [0] + n_features + [17.5]

fig = plt.figure()
ax = plt.subplot(111)

line_bin, = plt.plot(n_features, _bin, label='binary', linewidth=2)
line_rel, = plt.plot(n_features, rel, label='relative', linewidth=2)
line_tfidf, = plt.plot(n_features, tfidf, label='tfidf', linewidth=2)
line_tf, = plt.plot(n0_features,text_feats,label = 'text feats ref. line',color='k',linewidth=2)
line_naiv, = plt.plot(n0_features,naiv,linestyle = '--', label = 'modus ref. line',color='k',linewidth=2)
plt.legend(handles = [line_bin,line_rel,line_tfidf,line_tf, line_naiv], bbox_to_anchor=(1.55, 1.03))
plt.xlabel("Feature vector length [in thousands]")
plt.ylabel("Author classification accuracy (bayes. model)")
ax.set_xlim([-0.01,17.5])
ax.set_ylim([0,1])
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.show()
fig.savefig(os.path.join(EXPLORATION_FOLDER,'author_prediction_bow_length.pdf'),bbox_inches='tight')

author_counter = Counter(df_meta.author)
author_counter_counts=Counter([x[1] for x in author_counter.items()  if x[0] not in ['Anonymous','Unknown',None, 'Various']])
total = sum(author_counter_counts.values())
counts=[]
running_total = total
for i in range(1,21):
  counts+=[running_total]
  running_total -= author_counter_counts[i]

fig, ax = plt.subplots()
plt.plot(range(2,21),counts[1:],linewidth = 3, color = COLOR)
ax.set_xlim([0,20.02])
ax.set_ylim([0,4500])
plt.xlabel("Minimum number of books written")
plt.ylabel("Number of authors")
plt.show()
fig.savefig(os.path.join(EXPLORATION_FOLDER,'authors_with_n_books.pdf'),bbox_inches='tight')


####

n_trees = [1,10,100,500,1000]
perf = [0.241, 0.432, 0.648, 0.681, 0.691]

fig, ax = plt.subplots()
plt.plot(n_trees, perf,linewidth = 3, color = COLOR)
ax.set_xlim([1,1010])
ax.set_ylim([0,1])
plt.xscale('symlog')
plt.xlabel("Number of trees [logarithmic]")
plt.ylabel("Author prediction accuracy")
plt.show()
fig.savefig(os.path.join(EXPLORATION_FOLDER,'authors_random_text.pdf'),bbox_inches='tight')


### CORRPLOT

## TFIDF
from biokit import Corrplot
from models import load_fvs
#c = Corrplot(df.ix[:,:20])
#c.plot(method = 'circle', rotation = 45)

tfidf = load_fvs('tfidf',size='2k')
words_for_corr1 = ['also','various','number','similar','during','united','united states','states','you','go','look','thee','thou','hast','her','she','me','my','form','tell','ireland','irish','thus']
c2 = Corrplot(tfidf.ix[:,words_for_corr1])
fig, ax = plt.subplots()
c2.plot(method = 'circle', rotation = 45,fontsize = 12,ax=ax)
fig.savefig(os.path.join(EXPLORATION_FOLDER,'corr_plot_tfidf.pdf'),bbox_inches='tight')


binn = load_fvs('bin',size='2k')
words_for_corr2 = ['also','various','number','similar','during','united','united states','states','color','colour','favor','favour','you','go','look','thee','thou','hast','her','she','me','my','form','tell','ireland','irish','thus','honor','honour']

c3 = Corrplot(binn.ix[:,words_for_corr2])
fig, ax = plt.subplots()
c3.plot(method = 'circle', rotation = 45,fontsize = 12,ax=ax)
fig.savefig(os.path.join(EXPLORATION_FOLDER,'corr_plot_bin.pdf'),bbox_inches='tight')



# CORRPLOT text features
fig, ax = plt.subplots(figsize=(15,12))
ax.grid=False

text_feats = load_fvs('text_feats')
include_idx = range(27)
new_col = text_feats.columns.tolist()
for i in range(8,27):
  new_col[i] = pos_mapping[text_feats.columns[i]]
text_feats.columns = new_col

c4 = Corrplot(text_feats.ix[:,include_idx])
c4.plot(method = 'circle', rotation = 60,fig=fig,fontsize='x-large')
fig.savefig(os.path.join(EXPLORATION_FOLDER,'corr_plot_text_features.pdf'),bbox_inches='tight')

## SORTED CORRELATION
tfidf = load_fvs('tfidf',size='2k')
q = np.corrcoef(tfidf,rowvar=0)
qq = q.ravel()
qq_sorted = sorted(qq)
fig, ax = plt.subplots()
plt.plot(np.linspace(0,1,len(qq_sorted)),qq_sorted,linewidth=3,color=COLOR)
ax.set_ylim([-1,1])
plt.ylabel("Pearson correlation coefficient")
plt.xlabel("Proportion of correlation pairs (sorted)")
ax.axhline(0,color='k')
ax.axvline(0.01,color='k',linestyle = '-.',label='bottom 1 %')
ax.axvline(0.99,color='k',linestyle = '--', label = 'top 1 %')
plt.legend()
plt.show()
fig.savefig(os.path.join(EXPLORATION_FOLDER,'tfidf_sorted_corr.pdf'),bbox_inches='tight')

_bin = load_fvs('bin',size='2k')
q = np.corrcoef(_bin,rowvar=0)
qq = q.ravel()
qq_sorted = sorted(qq)
fig, ax = plt.subplots()
plt.plot(np.linspace(0,1,len(qq_sorted)),qq_sorted,linewidth=3,color=COLOR)
ax.set_ylim([-1,1])
plt.ylabel("Pearson correlation coefficient")
plt.xlabel("Proportion of correlation pairs (sorted)")
ax.axhline(0,color='k')
ax.axvline(0.01,color='k',linestyle = '-.',label='bottom 1 %')
ax.axvline(0.99,color='k',linestyle = '--', label = 'top 1 %')
plt.legend()
plt.show()
fig.savefig(os.path.join(EXPLORATION_FOLDER,'bin_sorted_corr.pdf'),bbox_inches='tight')