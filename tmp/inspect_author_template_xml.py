from docx import Document
p=r'C:\Users\11c16\OneDrive\Desktop\Deep-learning-capstome\tmp\docx_submission\template_base.docx'
d=Document(p)
for i in range(1,8):
 x=d.paragraphs[i]
 print('\nINDEX',i,'STYLE',x.style.name,'TEXT',repr(x.text))
 print(x._p.xml[:5500])
