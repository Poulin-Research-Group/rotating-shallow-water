      subroutine flux_ener_F(uvh, flux, ener, enst, Nx,Ny,params,inds)
      implicit none
      integer Nx, Ny
      double precision uvh(3*Ny,Nx)
      double precision flux(3*Ny,Nx), ener, enst

      double precision params(5), dx, dy, gp, f0, H0
      double precision h(Ny,Nx), U(Ny,Nx), V(Ny,Nx), B(Ny,Nx), q(Ny,Nx)
      integer inds(3), Iv_i, Iv_f, Ih_i

cf2py intent(in) :: uvh, params
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: flux, ener, enst

      dx = params(1)
      dy = params(2)
      gp = params(3)
      f0 = params(4)
      H0 = params(5)

      Iv_i = inds(1)
      Iv_f = inds(2)
      Ih_i = inds(3)

      call calc_h(uvh, h, Nx, Ny, H0, Ih_i)
      call calc_U(uvh, h, U, Nx, Ny)
      call calc_V(uvh, h, V, Nx, Ny, Iv_i, Iv_f)
      call calc_B(uvh, h, B, Nx, Ny, gp, Iv_i, Iv_f)
      call calc_q(uvh, h, q, Nx, Ny, dx, dy, f0, Iv_i, Iv_f)
      call calc_flux(q, U, V, B, flux, Nx, Ny, dx, dy)
      call calc_energy(uvh, h, ener, Nx, Ny, gp, Iv_i, Iv_f)
      call calc_enstrophy(h, q, enst, Nx, Ny)

      end


c     calculating the flux terms themselves ===================================
      subroutine calc_flux(q, U, V, B, flux, Nx, Ny, dx, dy)
      implicit none
      integer Nx, Ny
      double precision q(Ny,Nx), U(Ny,Nx), V(Ny,Nx), B(Ny,Nx)
      double precision flux(3*Ny,Nx)

      double precision dx, dy
      integer Ny2, r, c

      Ny2 = 2*Ny

cf2py intent(in) :: q, U, V, B, dx, dy
cf2py intent(in) :: Nx, Ny


c     calculating first term of flux array ==========================
      do c=1,Nx-1
        do r=2,Ny
          flux(r,c) = (q(r-1,c) * (V(r-1,c+1) + V(r-1,c)) 
     &              +  q(r,c)   * (V(r,c+1)   + V(r,c)))*0.25 
     &              - (B(r,c+1) - B(r,c)) / dx
        enddo

c       first row, all columns (except last)
        flux(1,c) = (q(Ny,c) * (V(Ny,c+1) + V(Ny,c))
     &            +  q(1,c)  * (V(1,c+1)  + V(1,c)))*0.25
     &            - (B(1,c+1) - B(1,c)) / dx

      enddo

c     last column, all rows (except first)
      do r=2,Ny 
        flux(r,Nx) = (q(r-1,Nx) * (V(r-1,1) + V(r-1,Nx)) 
     &             +  q(r,Nx)   * (V(r,1)   + V(r,Nx)))*0.25
     &             - (B(r,1) - B(r,Nx)) / dx
      enddo

c     first row, last column
      flux(1,Nx) = (q(Ny,Nx) * (V(Ny,1) + V(Ny,Nx))
     &           +  q(1,Nx)  * (V(1,1)  + V(1,Nx)))*0.25
     &           - (B(1,1) - B(1,Nx)) / dx


c     calculating second term of flux array =========================
      do c=2,Nx
        do r=1,Ny-1
          flux(Ny+r,c) = -(q(r,c-1) * (U(r+1,c-1) + U(r,c-1))
     &                +   q(r,c)   * (U(r+1,c)   + U(r,c)))*0.25 
     &                -  (B(r+1,c) - B(r,c)) / dy
        enddo

c       last row, all columns (except first)
        flux(Ny2,c) = -(q(Ny,c-1) * (U(1,c-1) + U(Ny,c-1))
     &               +   q(Ny,c)   * (U(1,c)   + U(Ny,c)))*0.25
     &               -  (B(1,c) - B(Ny,c)) / dy

      enddo

c     first column, all rows (except last)
      do r=1,Ny-1
        flux(Ny+r,1) = -(q(r,Nx) * (U(r+1,Nx) + U(r,Nx))
     &              +   q(r,1)  * (U(r+1,1)  + U(r,1)))*0.25
     &              -  (B(r+1,1) - B(r,1)) / dy
      enddo

c     last row, first column
      flux(Ny2,1) = -(q(Ny,Nx) * (U(1,Nx)  + U(Ny,Nx)) 
     &              +  q(Ny,1)  * (U(1,1)   + U(Ny,1)))*0.25
     &              - (B(1,1) - B(Ny,1)) / dy



c     calculating third term of flux array ==========================
      do c=2,Nx
        do r=2,Ny
          flux(Ny2+r,c) = (U(r,c-1) - U(r,c)) / dx
     &                  + (V(r-1,c) - V(r,c)) / dy
        enddo

c       first row, all columns (except first)
        flux(Ny2+1,c) = (U(1,c-1) - U(1,c)) / dx
     &                + (V(Ny,c)  - V(1,c)) / dy

      enddo

c     first column, all rows (except first)
      do r=2,Ny
        flux(Ny2+r,1) = (U(r,Nx)  - U(r,1)) / dx
     &                + (V(r-1,1) - V(r,1)) / dy
      enddo

c     first row, first column
      flux(Ny2+1,1) = (U(1,Nx) - U(1,1)) / dx
     &              + (V(Ny,1) - V(1,1)) / dy


      end


c     calculating height h ====================================================
      subroutine calc_h(uvh, h, Nx, Ny, H0, Ih_i)
      implicit none

      integer Nx, Ny
      double precision uvh(3*Ny, Nx), h(Ny, Nx)

      double precision H0
      integer Ih_i, r, c

cf2py intent(in) :: uvh, H0, Ih_i
cf2py intent(in) :: Nx, Ny

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
cf2py intent(in) :: Nx, Ny

      do c=1,Nx-1
        do r=1,Ny
          U(r, c) = 0.5*(h(r, c+1) + h(r,c)) * uvh(r, c)
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
cf2py intent(in) :: Nx, Ny

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
cf2py intent(in) :: Nx, Ny

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
     &        + 0.25*(uvh(1, 1)**2      + uvh(1, Nx)**2)
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
cf2py intent(in) :: Nx, Ny

      do c=1,Nx-1
        do r=1,Ny-1
          q(r, c) = ((uvh(r+Iv_i, c+1) - uvh(r+Iv_i, c)) / dx            ! rolling columns of V
     &            -  (uvh(r+1, c)      - uvh(r, c))      / dy            ! rolling rows of U
     &            +  f0)
     &            / (0.25*(h(r+1,c+1) + h(r+1,c) + h(r,c+1) + h(r,c)))
        enddo

c       last row of q, all columns (except last)
        q(Ny, c) = ((uvh(Iv_f, c+1) - uvh(Iv_f, c)) / dx
     &           -  (uvh(1, c)      - uvh(Ny, c))   / dy
     &           +  f0)
     &           / (0.25*(h(1,c+1) + h(1,c) + h(Ny,c+1) + h(Ny,c)))
      enddo

c     last column of q, all rows (except last)
      do r=1,Ny-1
        q(r, Nx) = ((uvh(r+Iv_i, 1) - uvh(r+Iv_i, Nx)) / dx
     &           -  (uvh(r+1, Nx)   - uvh(r, Nx))      / dy
     &           +  f0)
     &           / (0.25*(h(r+1,1) + h(r+1,Nx) + h(r,1) + h(r,Nx)))
      enddo

c     last column and row of q
      q(Ny, Nx) = ((uvh(Iv_f, 1) - uvh(Iv_f, Nx)) / dx
     &          -  (uvh(1, Nx)   - uvh(Ny, Nx))   / dy
     &          +  f0)
     &          / (0.25*(h(1,1) + h(1,Nx) + h(Ny,1) + h(Ny,Nx)))

      end

      

      subroutine calc_energy(uvh, h, energy, Nx, Ny, gp, Iv_i, Iv_f)
      implicit none
      integer Nx, Ny
      double precision uvh(3*Ny,Nx), h(Ny,Nx), energy

      double precision gp
      integer Iv_i, Iv_f, r, c

cf2py intent(in) :: uvh, h, gp, Iv_i, Iv_f
cf2py intent(in) :: Nx, Ny

      do c=2,Nx
        do r=2,Ny
          energy = energy + gp*h(r,c)**2 + 0.5*h(r,c)
     &           * (uvh(r,c)**2       + uvh(r,c-1)**2
     &           +  uvh(r+Iv_i, c)**2 + uvh(r+Iv_i-1, c)**2)
        enddo

c       first row, sort of.
        energy = energy + gp*h(1,c)**2 + 0.5*h(1,c)
     &         * (uvh(1,c)**2       + uvh(1,c-1)**2 
     &         +  uvh(Iv_i+1, c)**2 + uvh(Iv_f, c)**2)
      enddo

c     first column, sort of.
      do r=2,Ny
        energy = energy + gp*h(r,1)**2 + 0.5*h(r,1)
     &         * (uvh(r,1)**2       + uvh(r,Nx)**2
     &         +  uvh(r+Iv_i, 1)**2 + uvh(r+Iv_i-1, 1)**2)
      enddo

c     first row, first column... kinda.
      energy = energy + gp*h(1,1)**2 + 0.5*h(1,1) 
     &       * (uvh(1,1)**2       + uvh(1,Nx)**2 
     &       +  uvh(Iv_i+1, 1)**2 + uvh(Iv_f, 1)**2)

      energy = 0.5*energy / (Nx*Ny)

      end


      subroutine calc_enstrophy(h, q, enstrophy, Nx, Ny)
      implicit none
      integer Nx, Ny
      double precision h(Ny,Nx), q(Ny,Nx), enstrophy

      integer r, c

cf2py intent(in) :: h, q
cf2py intent(in) :: Nx, Ny

      do c=1,Nx-1
        do r=1,Ny-1
          enstrophy = enstrophy + q(r,c)**2 
     &              * (h(r+1,c+1) + h(r+1,c) + h(r,c+1) + h(r,c))
        enddo

c       last row
        enstrophy = enstrophy + q(Ny,c)**2
     &            * (h(1,c+1) + h(1,c) + h(Ny,c+1) + h(Ny,c))
      enddo

c     last column
      do r=1,Ny-1
        enstrophy = enstrophy + q(r,Nx)**2
     &            * (h(r+1,1) + h(r+1,Nx) + h(r,1) + h(r,Nx))
      enddo

c     last row and column
      enstrophy = enstrophy + q(Ny,Nx)**2
     &          * (h(1,1) + h(1,Nx) + h(Ny,1) + h(Ny,Nx))

      enstrophy = 0.125 * enstrophy / (Nx*Ny)

      end