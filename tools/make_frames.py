import numpy as np, glob, json, base64, io, math, sys
from PIL import Image
src=sys.argv[1]; out=sys.argv[2]
files=sorted(glob.glob(src+'/*.asc'), key=lambda s:s.split('_')[-1])
hdr={}
for line in open(files[0]).readlines()[:6]:
    k,v=line.split(); hdr[k.lower()]=float(v)
nc,nr=int(hdr['ncols']),int(hdr['nrows']); x0,y0,cs=hdr['xllcorner'],hdr['yllcorner'],hdr['cellsize']
W,S,E,N=x0,y0,x0+nc*cs,y0+nr*cs
grids=[]
for f in files:
    a=np.loadtxt(f,skiprows=6); a[a==-9999]=0; grids.append(a)
names=[f.split('_')[-1].split('.')[0] for f in files]
# colour ramp (depth m -> RGBA)
stops=[(0.0,(200,240,255,0)),(0.02,(170,225,250,110)),(0.15,(110,190,240,170)),(0.5,(50,130,210,205)),(1.0,(20,80,175,225)),(2.0,(10,40,120,240)),(3.5,(5,20,70,250))]
sd=np.array([s[0] for s in stops]); sc=np.array([s[1] for s in stops],float)
def ramp(d):
    return np.stack([np.interp(d,sd,sc[:,i]) for i in range(4)],-1)
# mercator row resample
my=lambda lat: math.log(math.tan(math.pi/4+math.radians(lat)/2))
mN,mS=my(N),my(S)
rows=np.arange(nr)
lat_t=np.degrees(2*np.arctan(np.exp(mN-(rows+0.5)/nr*(mN-mS)))-math.pi/2)
srcrow=np.clip(((N-lat_t)/cs).astype(int),0,nr-1)
frames=[]
for n,g in zip(names,grids):
    rgba=ramp(np.round(g[srcrow]/0.02)*0.02).astype(np.uint8); rgba[g[srcrow]<0.01,3]=0
    im=Image.fromarray(rgba,'RGBA').quantize(256,method=Image.Quantize.FASTOCTREE); im.save(f'{out}/frames/flood_{n}.png',optimize=True)
    b=io.BytesIO(); im.save(b,'PNG',optimize=True); frames.append('data:image/png;base64,'+base64.b64encode(b.getvalue()).decode())
# 3D cells, block 3
k=3; R,C=nr//k,nc//k
blk=[g[:R*k,:C*k].reshape(R,k,C,k).mean((1,3)) for g in grids]
mx=np.max(blk,0); idx=np.argwhere(mx>0.05)
lon=W+(idx[:,1]+0.5)*k*cs; lat=N-(idx[:,0]+0.5)*k*cs
cells={'pos':np.round(np.stack([lon,lat],1).ravel(),6).tolist(),
       'depth':[np.round(b[idx[:,0],idx[:,1]],2).tolist() for b in blk],
       'cellM':k*cs*111320*math.cos(math.radians((S+N)/2))}
data={'bounds':[W,S,E,N],'names':names,'frames':frames,'cells':cells,'maxDepth':float(max(g.max() for g in grids)),
      'stats':[{'wetKm2':float((g>0.05).sum()*(cs*111320*math.cos(math.radians((S+N)/2)))*(cs*110574)/1e6),'max':float(g.max())} for g in grids]}
open(f'{out}/data.js','w').write('window.FLOOD_DATA='+json.dumps(data,separators=(',',':'))+';')
json.dump({'bounds_WSEN':[W,S,E,N],'corners_for_maplibre_image_source':[[W,N],[E,N],[E,S],[W,S]],'frames':[f'flood_{n}.png' for n in names],'note':'PNGs resampled to Web Mercator rows; depth colour ramp in metres'},open(f'{out}/frames/bounds.json','w'),indent=1)
print(len(idx),'cells', [d['wetKm2'] for d in data['stats']], W,S,E,N)
