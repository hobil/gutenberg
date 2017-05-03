import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import os
from metadata import load_metadata

df_meta = load_metadata()
EXPLORATION_FOLDER='exploration'
plt.rcParams['figure.titlesize']='xx-large'  # fontsize of the axes title
plt.rcParams['axes.labelsize']='xx-large' # fontsize of the x any y labels
plt.rcParams['xtick.labelsize']='xx-large'
plt.rcParams['ytick.labelsize']='xx-large'
plt.rcParams['axes.ymargin']=1
plt.rcParams['axes.grid']=True
plt.rcParams['legend.fontsize']='xx-large'
COLOR="#1f7bb4"
COLOR2="g"
COLOR3="r"

########   AUTHORS   ###########
author_counter=Counter(df_meta.author)
print("\n########   AUTHORS   ###########")
print("There are %d author tags.\n"%len(author_counter))
i=0

# show top 20 authors
print("Top 20 author tags:")
for x in author_counter.most_common()[:20]:
  print(x)
print()

# what lcc tags do the Various have
various=df_meta[df_meta.author=='Various'].named_LCC
lccs=dict()
for named_LCC in various:
  #for x in set([x[:1] for x in named_LCC]):
  for x in set(named_LCC):
    if x not in lccs:
      lccs[x]=1
    else:
      lccs[x]+=1
sorted_lccs = sorted(lccs.items(), key=lambda x: x[1],reverse=True)
print("Top Various LCC classes:")
for x in sorted_lccs[:10]:
  print(x)
print()
  
# what lcc tags do the Anonymous have
various=df_meta[df_meta.author=='Anonymous'].named_LCC
lccs=dict()
for named_LCC in various:
  #for x in set([x[:1] for x in named_LCC]):
  for x in set(named_LCC):
    if x not in lccs:
      lccs[x]=1
    else:
      lccs[x]+=1
sorted_lccs = sorted(lccs.items(), key=lambda x: x[1],reverse=True)
print("Top Anonymous LCC classes:")
for x in sorted_lccs[:10]:
  print(x)
print()
  
# df_meta[df_meta.LCC.apply(lambda x: ('DA' in x)) & (df_meta.author=="Anonymous")]
# df_meta[df_meta.author=='Various'].groupby('lcc_class').count().LCC


# exlude Various, Anonymous, None and Unknown
AUTHORS_TO_EXCLUDE=['Various',None,'Anonymous','Unknown']
no_of_books_by_author=np.array([c for a, c in author_counter.items() if a not in AUTHORS_TO_EXCLUDE])
# authors with one book
no_of_books_by_author_counter=Counter(no_of_books_by_author)

print(no_of_books_by_author_counter.most_common(10))

fig = plt.figure()
axes=plt.gca()
axes.set_ylim([0,50000])
axes.set_xlim([0,225])
plt.yscale('symlog')
#plt.xscale('symlog')
plt.hist(no_of_books_by_author,bins=range(0,230,10),color=COLOR2)
#fig.suptitle('test title')
plt.xlabel('Documents per Author')
plt.ylabel('No. of Authors (Logarithmic)')
i=1
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_docs_per_author.pdf'%i),bbox_inches='tight')



no_of_books, no_of_authors = zip(*sorted(no_of_books_by_author_counter.items(),key = lambda x: x[0]))
no_of_books = np.array(no_of_books)
no_of_authors = np.array(no_of_authors)

fig = plt.figure()
axes=plt.gca()
axes.set_ylim([9000,15000])
axes.set_xlim([0,25])
#plt.yscale('symlog')
#plt.xscale('symlog')
mask = no_of_books <= 25
plt.plot(no_of_books[mask],np.cumsum(no_of_authors[mask]),color=COLOR2, linewidth = 3)
#fig.suptitle('test title')
plt.xlabel('No. of Documents Written by Author')
plt.ylabel('No. of Authors who wrote max. n documents')
i=1
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_max_n_books.pdf'%i),bbox_inches='tight')





#########     YEAR   ###############
years=df_meta[df_meta.authoryearofbirth.notnull()].authoryearofbirth
print("There are %d year tags.\n"%len(years))

fig = plt.figure()
axes=plt.gca()
axes.set_ylim([0,50000])
axes.set_xlim([-850,2050])
plt.yscale('symlog')
plt.hist(years,bins=range(-800,2050,100),color=COLOR)
plt.xlabel('Author\'s Birth Year')
plt.ylabel('Number of Documents (Logarithmic)')
i=2
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_docs_per_year.pdf'%i),bbox_inches='tight')


years_counter = Counter(years)
year, no_of_documents = zip(*sorted(years_counter.items(),key = lambda x: x[0]))

fig = plt.figure()
axes=plt.gca()
#axes.set_ylim([9000,15000])
#axes.set_xlim([0,25])
plt.yscale('symlog')
#plt.xscale('symlog')
#mask = no_of_books < 25
plt.plot(year,np.cumsum(no_of_documents),color=COLOR, linewidth = 3)
#fig.suptitle('test title')
#plt.xlabel('Number of Documents Written by Author')
#plt.ylabel('No. of Authors who wrote max. n documents')
i=1



eras=list(df_meta[df_meta.epoch_names.notnull()].epoch_names)
labels=['Ancient', 'Medieval', 'Renaissance', 'Baroque', 'Enlightenment', 'Romanticism', 'Realism', 'Late 19th', 'Modern']
labels_short=['ANC', 'MED', 'REN', 'BAR', 'ENL', 'ROM', 'REA', 'L19', 'MOD']
counts=[eras.count(l) for l in labels]
index=np.arange(len(labels))
bar_width=0.35
fig = plt.figure()
axes=plt.gca()
axes.set_ylim([0,15000])
plt.bar(index,counts,color=COLOR)
plt.xticks(index+bar_width,labels_short)
plt.xlabel('Epoch')
plt.ylabel('Number of Documents')
i=3
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_docs_per_era.pdf'%i),bbox_inches='tight')




#########     LCC   ###############
print("#########     LCC   ###############")
total=len(df_meta)
sizes=Counter(df_meta.LCC.apply(lambda x: len(x)))
for s in sizes.items():
  print("%d tags -- %d documents, %.1f %%"%(s[0],s[1],100*s[1]/total))

# lcc parent class except for P
lcc_named_list1=[]
for l in df_meta.named_LCC_1:
    lcc_named_list1+=list(l)
lcc_named_list_counter1=Counter(lcc_named_list1)
for x in lcc_named_list_counter1.most_common(10):
  print("%d, %s"%(x[1],x[0]))

# whole lcc class
lcc_named_list=[]
for l in df_meta.named_LCC:
    lcc_named_list+=list(l)
lcc_named_list_counter=Counter(lcc_named_list)
for x in lcc_named_list_counter.most_common(10):
  print("%d, %s"%(x[1],x[0]))


#########     LCSH   ###############

# tags per document
print("#########     LCSH   ###############")
total=len(df_meta)
lcsh_lengths=df_meta.subjects.apply(lambda x: len(x))
for s in Counter(lcsh_lengths).items():
  print("%d tags -- %d documents, %.1f %%"%(s[0],s[1],100*s[1]/total))

fig = plt.figure()
axes=plt.gca()
axes.set_ylim([0,16000])
axes.set_xlim([0,20])
#plt.bar(index,counts,color=COLOR)
plt.hist(lcsh_lengths,bins=range(0,20),color=COLOR)
plt.xlabel('Number of LCSH tags')
plt.ylabel('Number of Documents')
i=4
fig.savefig(os.path.join(EXPLORATION_FOLDER,'%d_lcshs_per_doc.pdf'%i),bbox_inches='tight')

# documents per tag
lcsh=set.union(*[s for s in df_meta.subjects])
print("There are %d unique lcsh tags."%len(lcsh))

lcsh_list=[]
for l in df_meta.subjects:
    lcsh_list+=list(l)
lcsh_counter=Counter(lcsh_list)

for x in lcsh_counter.most_common(10):
  print("%d, %s"%(x[1],x[0]))

counts=[x for x in lcsh_counter.values()]
lcsh_counter_counter=Counter(counts)

# simple lcsh tags
lcsh_simple=set.union(*[s for s in df_meta.lcsh_simple])
print("There are %d unique lcsh_simple tags."%len(lcsh_simple))

lcsh_simple_list=[]
for l in df_meta.query('PRSZ').lcsh_simple:
    lcsh_simple_list+=list(l)
lcsh_simple_counter=Counter(lcsh_simple_list)

for x in lcsh_simple_counter.most_common(30):
  print(x[0])
    
counts_simple=[x[1] for x in lcsh_simple_counter.items()]
lcsh_simple_counter_counter=Counter(counts_simple)

