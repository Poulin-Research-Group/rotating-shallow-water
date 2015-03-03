c     calculating height h ====================================================
      subroutine calc_h(uvh, h, Nx, Ny, H0, Ih_i)
      implicit none

      integer Nx, Ny
      double precision uvh(3*Ny, Nx), h(Ny, Nx)

      double precision H0
      integer Ih_i, r, c

cf2py intent(in) :: uvh, H0, Ih_i
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: h

      do c=1,Nx
        do r=1,Ny
          h(r, c) = H0 + uvh(r+Ih_i, c)
        enddo
      enddo

      end


c     calculating zonal mass flux U ===========================================
      subroutine calc_U(uvh, h, U, Nx, Ny)
      implicit none

      integer Nx, Ny
      double precision uvh(3*Ny, Nx), h(Ny, Nx), U(Ny, Nx)

      integer r, c

cf2py intent(in) :: uvh, h
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: U

      do c=1,Nx-1
        do r=1,Ny
          U(r, c) = 0.5*(h(r, c+1) + h(r,c)) * uvh(r, c)
          !            roll
        enddo
      enddo

      do r=1,Ny
        U(r, Nx) = 0.5*(h(r, 1) + h(r, Nx)) * uvh(r, Nx)
      enddo

      end


c     calculating meridonal mass flux V =======================================
      subroutine calc_V(uvh, h, V, Nx, Ny, Iv_i, Iv_f)
      implicit none

      integer Nx, Ny
      double precision uvh(3*Ny, Nx), h(Ny, Nx), V(Ny, Nx)

      integer Iv_i, Iv_f, r, c

cf2py intent(in) :: uvh, h, Iv_i, Iv_f
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: V

      do c=1,Nx
        do r=1,Ny-1
          V(r, c) = 0.5*(h(r+1, c) + h(r,c)) * uvh(r+Iv_i, c)
        enddo

        V(Ny, c) = 0.5*(h(1, c) + h(Ny, c)) * uvh(Iv_f, c)
      enddo


      end


c     calculating Bernoulli function B ========================================
      subroutine calc_B(uvh, h, B, Nx, Ny, gp, Iv_i, Iv_f)
      implicit none
      integer Nx, Ny
      double precision uvh(3*Ny, Nx), h(Ny, Nx), B(Ny, Nx)

      double precision gp
      integer Iv_i, Iv_f, r, c

cf2py intent(in) :: uvh, h, gp, Iv_i, Iv_f
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: B

      do c=2,Nx
        do r=2,Ny
          B(r, c) = gp*h(r, c)
     &            + 0.25*(uvh(r, c)**2      + uvh(r, c-1)**2)     
     &            + 0.25*(uvh(r+Iv_i, c)**2 + uvh(r+Iv_i-1, c)**2)
        enddo

        B(1, c) = gp*h(1, c)
     &          + 0.25*(uvh(1, c)**2      + uvh(1, c-1)**2)
     &          + 0.25*(uvh(Iv_i+1, c)**2 + uvh(Iv_f, c)**2)
      enddo

      do r=2,Ny
        B(r, 1) = gp*h(r, 1)
     &          + 0.25*(uvh(r, 1)**2      + uvh(r, Nx)**2)
     &          + 0.25*(uvh(r+Iv_i, 1)**2 + uvh(r+Iv_i-1, 1)**2)
      enddo

      B(1, 1) = gp*h(1, 1)
     &        + 0.25*(uvh(1, 1)**2 + uvh(1, Nx)**2)
     &        + 0.25*(uvh(Iv_i+1, 1)**2 + uvh(Iv_f, 1)**2)

      end


c     calculating potential vorticity q =======================================
      subroutine calc_q(uvh, h, q, Nx, Ny, dx, dy, f0, Iv_i, Iv_f)
      implicit none
      integer Nx, Ny
      double precision uvh(3*Ny, Nx), h(Ny, Nx), q(Ny, Nx)

      double precision dx, dy, f0
      integer Iv_i, Iv_f, r, c

cf2py intent(in) :: uvh, h, dx, dy, f0, Iv_i, Iv_f
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: q

      do c=1,Nx-1
        do r=1,Ny-1
          q(r, c) = ((uvh(r+Iv_i, c) + uvh(r+Iv_i, c+1)) / dx
     &            +  (uvh(r, c)      + uvh(r+1, c)) / dy
     &            +  f0)
     &            / (0.25*(h(r+1,c+1) + h(r+1,c) + h(r,c+1) + h(r,c)))
        enddo

c       last row of q, all columns (except last)
        q(Ny, c) = ((uvh(Iv_f, c) + uvh(Iv_f, c+1)) / dx
     &           +  (uvh(Ny, c)   + uvh(1, c)) / dy
     &           +  f0)
     &           / (0.25*(h(1,c+1) + h(1,c) + h(Ny,c+1) + h(r,c)))
      enddo

c     last column of q, all rows (except last)
      do r=1,Ny-1
        q(r, Nx) = ((uvh(r+Iv_i, Nx) + uvh(r+Iv_i, 1)) / dx
     &           +  (uvh(r, Nx)      + uvh(r+1, Nx)) / dy
     &           +  f0)
     &           / (0.25*(h(r+1,1) + h(r+1,Nx) + h(r,1) + h(r,c)))
      enddo

c     last column and row of q
      q(Ny, Nx) = ((uvh(Iv_f, Nx) + uvh(Iv_i+1, 1)) / dx
     &          +  (uvh(Ny, Nx)   + uvh(Ny, 1)) / dy
     &          +  f0)
     &          / (0.25*(h(1,1) + h(1,Nx) + h(Ny,1) + h(Ny,Nx)))

      end

      
c     calculating first term of flux array ====================================      
      subroutine calc_flux_1(q, V, B, flux_1, Nx, Ny, dx)
      implicit none
      integer Nx, Ny
      double precision q(Ny,Nx), V(Ny,Nx), B(Ny,Nx), flux_1(Ny,Nx)

      double precision dx
      integer r, c

cf2py intent(in) :: q, V, B, dx
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: flux_1

      do c=1,Nx-1
        do r=2,Ny
          flux_1(r,c) = (q(r-1,c) * (V(r-1,c+1) + V(r-1,c)) 
     &                +  q(r,c) * (V(r,c+1) + V(r,c)))*0.25 
     &                - (B(r,c+1) - B(r,c)) / dx
        enddo

c       first row, all columns (except last)
        flux_1(1,c) = (q(Ny,c) * (V(Ny,c+1) + V(Ny,c))
     &              +  q(1,c) * (V(r,c+1) + V(r,c)))*0.25
     &              - (B(1,c+1) - B(1,c)) / dx

      enddo

c     last column, all rows (except first)
      do r=2,Ny 
        flux_1(r,Nx) = (q(r-1,Nx) * (V(r-1,1) + V(r-1,c)) 
     &               +  q(1,c) * (V(r,1) + V(r,c)))*0.25
     &               - (B(r,1) - B(r,c)) / dx
      enddo

c     first row, last column
      flux_1(1,Nx) = (q(Ny,c) * (V(Ny,1) + V(Ny,Nx)) 
     &             +  q(1,Nx) * (V(r,1) + V(1,Nx)))*0.25
     &             - (B(1,1) - B(1,Nx)) / dx

      end