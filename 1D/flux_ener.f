      subroutine flux_ener(uvh,flux,menerg,menstr,Nx,pfort)

c     FJP: add energy, enstrophy and remove flux from call line
c     FJP: first test to see if this works!
      implicit none
      integer Nx,j
      double precision uvh(3,Nx), flux(3,Nx)
      double precision pfort(4)

      double precision dx, gp, f0, H0
      double precision h(Nx), U(Nx), V(Nx), B(Nx), q(Nx)
      double precision energy(Nx), enstrophy(Nx)
      double precision menerg, menstr

cf2py intent(in) :: uvh
cf2py intent(in) :: parms
cf2py intent(hide) :: Nx
cf2py intent(out) :: flux
cf2py intent(out) :: menerg
cf2py intent(out) :: menstr

c     import parameters
      dx = pfort(1)
      gp = pfort(2)
      f0 = pfort(3)
      H0 = pfort(4)

c     compute height and meridonal mass flux: h, V
      DO j=1,Nx
         h(j)= H0 + uvh(3,j)
         V(j)= uvh(2,j)*h(j)
      END DO

c     compute zonal mass flux: U
      DO j=1,Nx-1
         U(j)= 0.5*(h(j+1) + h(j))*uvh(1,j)
      END DO
      U(Nx)= 0.5*(h(1) + h(Nx))*uvh(1,Nx)

c     compute Bernoulli function: B
      DO j=2,Nx
         B(j)= gp*h(j)+0.5*(0.5*(uvh(1,j)**2+uvh(1,j-1)**2)+uvh(2,j)**2)
      END DO
      B(1)= gp*h(1)+0.5*(0.5*(uvh(1,1)**2+uvh(1,Nx)**2)+uvh(2,1)**2)

c     compute Potentail Vorticity: q
      DO j=1,Nx-1
         q(j)= ((uvh(2,j+1) - uvh(2,j))/dx + f0)/(0.5*(h(j) + h(j+1)))
      END DO
      q(Nx)= ((uvh(2,1) - uvh(2,Nx))/dx + f0)/(0.5*(h(Nx) + h(1)))

c     compute flux for dU/dt
      DO j=1,Nx-1
         flux(1,j)= q(j)*0.5*(V(j+1) + V(j)) - (B(j+1) - B(j))/dx
      END DO
      flux(1,Nx)= q(Nx)*0.5*(V(1) + V(Nx)) - (B(1) - B(Nx))/dx

c     computes fluxes for dV/dt and dh/dt
      DO j=2,Nx
         flux(2,j)= -0.5*(q(j)*U(j) + q(j-1)*U(j-1))
         flux(3,j)= -(U(j) - U(j-1))/dx
      END DO
      flux(2,1)= -0.5*(q(1)*U(1) + q(Nx)*U(Nx))
      flux(3,1)= -(U(1) - U(Nx))/dx

c     compute energy
      DO j=2,Nx
         energy(j)= gp*h(j)+0.5*(uvh(1,j)**2+uvh(1,j-1)**2)+uvh(2,j)**2
         energy(j) = 0.5*h(j)*energy(j)
      END DO
      energy(1)= gp*h(1)+0.5*(uvh(1,1)**2+uvh(1,Nx)**2)+uvh(2,1)**2
      energy(1) = 0.5*h(1)*energy(1)

c     compute enstrophy
      DO j=1,Nx-1
         enstrophy(j)= 0.25*(h(j+1)+h(j))*q(j)**2
      END DO
      enstrophy(Nx)= 0.25*(h(1)+h(Nx))*q(Nx)**2

c     compute mean
      menerg = 0.0
      menstr = 0.0
      DO j=1,Nx
         menerg = menerg + energy(j)
         menstr = menstr + enstrophy(j)
      END DO
      
      menerg = menerg/Nx
      menstr = menstr/Nx
      
      end
      
