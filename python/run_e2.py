from allorders_2 import OrderE2, N3
from plots_allorders import plotvector2, plot_eigenfunctions, plotN3

def minmax(field):
    print("{0}: min = {1:10.5e} max = {2:10.5e}".format(field.name,field['g'].min(), field['g'].max()))

e2 = OrderE2()
# plotvector2(e2,1e-3,savetitle='V21_norm_false',
#             umax20=200,umin20=-200,Amax20=0.4,Amin20=-0.4,
#             psimax21=0,psimin21=-30,umax21=50,umin21=-100,
#             Amax21=20,Amin21=-40,Bmax21=-130,Bmin21=-170,
#             psimax22=5,psimin22=-5,umax22=20,umin22=-20,
#             Amax22=1,Amin22=-1,Bmax22=2,Bmin22=-2)
plotvector2(e2,1e-3,savetitle='V21_norm_false',
            umax20=200,umin20=-200,Amax20=0.4,Amin20=-0.4,
            psimax21=0,psimin21=-3,umax21=5,umin21=-10,
            Amax21=2,Amin21=-4,Bmax21=-13,Bmin21=-17,
            psimax22=5,psimin22=-5,umax22=20,umin22=-20,
            Amax22=1,Amin22=-1,Bmax22=2,Bmin22=-2)

for f in [e2.psi22,e2.u22,e2.A22,e2.B22]:
    minmax(f)

plot_eigenfunctions(e2.o1,savename='V11.png')
plot_eigenfunctions(e2.ah,savename='AH.png')

n3 = N3(o1=e2.o1,o2=e2)
plotN3(n3, 1e-3,psimax31=2000,psimin31=-2000,umax31=300,umin31=-300, Amax31=10,Amin31=-10,Bmax31=200,Bmin31=-200)
