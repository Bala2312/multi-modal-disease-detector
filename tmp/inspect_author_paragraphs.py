from docx import Document
p=r'C:\Users\11c16\OneDrive\Desktop\Deep-learning-capstome\tmp\docx_submission\template_base.docx'
d=Document(p)
for i in (4,5,6,7):
 x=d.paragraphs[i]
 print('INDEX',i,'STYLE',x.style.name,'ALIGN',x.alignment,'TEXT',repr(x.text))
 print('pPr',x._p.pPr.xml if x._p.pPr is not None else None)
 print('runs',[(r.text, r.font.size.pt if r.font.size else None) for r in x.runs[:10]])
