"""GeoIA - Clasificador educativo de figuras 2D con PyTorch.
Ejecutar: python geoia.py --train ; python geoia.py --predict ruta_imagen.png
"""
import argparse, json, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASES=['circulo','cuadrado','triangulo','rectangulo']
SIZE=64
SEED=42

def figura(clase, rng):
    im=Image.new('L',(SIZE,SIZE),0)
    d=ImageDraw.Draw(im)
    x=rng.randint(8,18); y=rng.randint(8,18)
    if clase==0:
        r=rng.randint(17,22); cx=rng.randint(29,35);cy=rng.randint(29,35)
        d.ellipse((cx-r,cy-r,cx+r,cy+r),fill=255)
    elif clase==1:
        lado=rng.randint(30,40); d.rectangle((x,y,x+lado,y+lado),fill=255)
    elif clase==2:
        top=(rng.randint(27,37),rng.randint(7,15))
        d.polygon([top,(rng.randint(7,17),rng.randint(46,56)),(rng.randint(47,57),rng.randint(46,56))],fill=255)
    else:
        w=rng.randint(38,48);h=rng.randint(19,28)
        d.rectangle((rng.randint(7,12),rng.randint(17,25),0,0),fill=0) if False else None
        x=rng.randint(7,12);y=rng.randint(17,25)
        d.rectangle((x,y,x+w,y+h),fill=255)
    a=np.array(im,dtype=np.float32)/255
    a=np.clip(a + rng.normalvariate(0,0.018)*np.random.default_rng(rng.randint(0,2**32-1)).standard_normal(a.shape),0,1)
    return a.astype('float32')

def dataset(por_clase,seed):
    rng=random.Random(seed)
    xs=[];ys=[]
    for c in range(4):
        for _ in range(por_clase):
            xs.append(figura(c,rng));ys.append(c)
    permutation=np.random.default_rng(seed).permutation(len(ys))
    return torch.from_numpy(np.stack(xs)[permutation,None]),torch.tensor(np.array(ys)[permutation],dtype=torch.long)

class Modelo(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
          nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),
          nn.Linear(16*16*16,32),nn.ReLU(),nn.Dropout(0.2),nn.Linear(32,4))
    def forward(self,x):return self.net(x)

def entrenar(output='resultados',epocas=10,por_clase=180):
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    xt,yt=dataset(por_clase,101)
    xv,yv=dataset(max(30,por_clase//4),202)
    xp,yp=dataset(max(30,por_clase//4),303)
    dl=DataLoader(TensorDataset(xt,yt),batch_size=64,shuffle=True)
    model=Modelo(); opt=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
    crit=nn.CrossEntropyLoss();hist=[];best=float('inf');best_epoch=0
    for epoch in range(epocas):
        model.train(); losses=[]
        for x,y in dl:
            opt.zero_grad();loss=crit(model(x),y);loss.backward();opt.step();losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            logits=model(xv); val_loss=crit(logits,yv).item();acc=(logits.argmax(1)==yv).float().mean().item()
        hist.append({'epoca':epoch+1,'perdida_entrenamiento':round(float(np.mean(losses)),5),'perdida_validacion':round(val_loss,5),'exactitud_validacion':round(acc,5)})
        print(f'Epoca {epoch+1}/{epocas}: val_acc={acc:.3f}; val_loss={val_loss:.4f}',flush=True)
        if val_loss<best:
            best=val_loss;best_epoch=epoch+1;torch.save(model.state_dict(),out/'modelo_geoia.pt')
    model.load_state_dict(torch.load(out/'modelo_geoia.pt',map_location='cpu',weights_only=True));model.eval()
    with torch.no_grad(): preds=model(xp).argmax(1).numpy()
    truth=yp.numpy();score=float((preds==truth).mean())
    metrics={'exactitud_prueba':score,'aciertos':int((preds==truth).sum()),'muestras_prueba':len(truth),'mejor_epoca':best_epoch,'historial':hist,'reporte':classification_report(truth,preds,target_names=CLASES,output_dict=True,zero_division=0),'matriz_confusion':confusion_matrix(truth,preds).tolist(),'particiones':{'entrenamiento':len(yt),'validacion':len(yv),'prueba':len(yp)}}
    (out/'metricas.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7,4));ax.plot([v['epoca'] for v in hist],[v['exactitud_validacion'] for v in hist],marker='o');ax.set(xlabel='Época',ylabel='Exactitud de validación',title='Evolución del modelo GeoIA',ylim=(0,1.05));ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/'curva_validacion.png',dpi=160);plt.close(fig)
    cm=np.array(metrics['matriz_confusion']);fig,ax=plt.subplots(figsize=(6,5));ax.imshow(cm,cmap='Blues');ax.set_xticks(range(4),CLASES,rotation=25);ax.set_yticks(range(4),CLASES);ax.set(xlabel='Predicción',ylabel='Real',title='Matriz de confusión (prueba)')
    for i in range(4):
        for j in range(4):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.tight_layout();fig.savefig(out/'matriz_confusion.png',dpi=160);plt.close(fig)
    demo=np.stack([figura(i,random.Random(500+i)) for i in range(4)])
    fig,axes=plt.subplots(1,4,figsize=(8,2.5))
    for i,ax in enumerate(axes):ax.imshow(demo[i],cmap='gray',vmin=0,vmax=1);ax.set_title(CLASES[i]);ax.axis('off')
    fig.tight_layout();fig.savefig(out/'muestras.png',dpi=160);plt.close(fig)
    Image.fromarray((demo[0]*255).astype('uint8')).save(out/'imagen_ejemplo_circulo.png')
    print(f'PRUEBA: {metrics["aciertos"]}/{len(truth)} = {score:.2%}',flush=True)
    return metrics

def predecir(ruta,modelo='resultados/modelo_geoia.pt'):
    net=Modelo();net.load_state_dict(torch.load(modelo,map_location='cpu',weights_only=True));net.eval()
    im=Image.open(ruta).convert('L').resize((SIZE,SIZE))
    x=np.asarray(im,dtype=np.float32)/255
    with torch.no_grad(): p=torch.softmax(net(torch.tensor(x)[None,None]),dim=1)[0].numpy()
    print('Predicción:',CLASES[int(p.argmax())]);print('Probabilidades:',{name:round(float(prob),4) for name,prob in zip(CLASES,p)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--predict');ap.add_argument('--output',default='resultados');ap.add_argument('--epocas',type=int,default=10);ap.add_argument('--por-clase',type=int,default=180)
    a=ap.parse_args()
    if a.train:entrenar(a.output,a.epocas,a.por_clase)
    elif a.predict:predecir(a.predict,str(Path(a.output)/'modelo_geoia.pt'))
    else:ap.print_help()
