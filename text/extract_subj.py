# %%
import importlib.util 
spec = importlib.util.spec_from_file_location('dbpediaNEL.DBPediaEntityExtractor', '../knowledgeGraph/kgqalib/dbpediaNEL/DBPediaEntityExtractor.py')
DBPediaEntityExtractor = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = DBPediaEntityExtractor 
spec.loader.exec_module(DBPediaEntityExtractor)
# %%
dbpee = DBPediaEntityExtractor.DBPediaEntityExtractor(mode="custom")
nlp = dbpee.nlp
# %%
# q = "What did Barack Obama's wife say about Adolf Hitler's Germany?"
q = "What did the child of Barack Obama say about Vladimir Putin in Germany?"
# q = "How old is Barack Obama's child?"
# q = "Who is the wife of Barack Obama?"
# q = "How old is Barack Obama's child born in Africa?"
# q = "When did Apple annonunce the first IPhone?"
# q = "Who was the first explorator during the Renaissance who discovered India?"
doc = nlp(q, disable=['dbpedia_spotlight'])
for chunk in doc.noun_chunks:
    print(chunk.text, '|', chunk.root.dep_,)
print('*'*25)
print('MAIN:', dbpee.extractMain(q))
print('LAST:', dbpee.extractLast(q))
print('ALL:', dbpee.extract(q))

# %%
