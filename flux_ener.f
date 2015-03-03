c     calculating height h ====================================================
      subroutine calc_h(uvh, h, Nx, Ny, H0, Ih_i)
      implicit none

      integer Nx, Ny
      double precision uvh(3*Ny, Nx), h(Ny, Nx)

      double precision H0
      integer Ih_i, i, j

cf2py intent(in) :: uvh, H0, Ih_i
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: h

      do j=1,Nx
        do i=1,Ny
          h(i, j) = H0 + uvh(i+Ih_i, j)
        enddo
      enddo

      end


c     calculating zonal mass flux U ===========================================
      subroutine calc_U(uvh, h, U, Nx, Ny)
      implicit none

      integer Nx, Ny
      double precision uvh(3*Ny, Nx), h(Ny, Nx), U(Ny, Nx)

      integer i, j

cf2py intent(in) :: uvh, h
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: U

      do j=1,Nx-1
        do i=1,Ny
          U(i, j) = 0.5*(h(i, j+1) + h(i,j))  * uvh(i, j)
          !            roll
        enddo
      enddo

      do i=1,Ny
        U(i, Nx) = 0.5*(h(i, 1) + h(i, Nx)) * uvh(i, Nx)
      enddo

      end


c     calculating meridonal mass flux V =======================================
      subroutine calc_V(uvh, h, V, Nx, Ny, Iv_i, Iv_f)
      implicit none

      integer Nx, Ny
      double precision uvh(3*Ny, Nx), h(Ny, Nx), V(Ny, Nx)

      integer Iv_i, Iv_f, i, j

cf2py intent(in) :: uvh, h, Iv_i, Iv_f
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: V

      do j=1,Nx
        do i=1,Ny-1
          V(i, j) = 0.5*(h(i+1, j) + h(i,j)) * uvh(i+Iv_i, j)
        enddo

        V(Ny, j) = 0.5*(h(1, j) + h(Ny, j)) * uvh(Iv_f, j)
      enddo


      end


c     calculating Bernoulli function B ========================================
      subroutine calc_B(uvh, h, B, Nx, Ny, gp, Iv_i, Iv_f)
      implicit none
      integer Nx, Ny
      double precision uvh(3*Ny, Nx), h(Ny, Nx), B(Ny, Nx)

      double precision gp
      integer Iv_i, Iv_f, i, j

cf2py intent(in) :: uvh, h, gp, Iv_i, Iv_f
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: B

      do j=2,Nx
        do i=2,Ny
          B(i, j) = gp*h(i, j)
     &            + 0.25*(uvh(i, j)**2      + uvh(i, j-1)**2)     
     &            + 0.25*(uvh(i+Iv_i, j)**2 + uvh(i+Iv_i-1, j)**2)
        enddo

        B(1, j) = gp*h(1, j)
     &          + 0.25*(uvh(1, j)**2      + uvh(1, j-1)**2)
     &          + 0.25*(uvh(Iv_i+1, j)**2 + uvh(Iv_f, j)**2)
      enddo

      do i=2,Ny
        B(i, 1) = gp*h(i, 1)
     &          + 0.25*(uvh(i, 1)**2      + uvh(i, Nx)**2)
     &          + 0.25*(uvh(i+Iv_i, 1)**2 + uvh(i+Iv_i-1, 1)**2)
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
      integer Iv_i, Iv_f, i, j

cf2py intent(in) :: uvh, h, dx, dy, f0, Iv_i, Iv_f
cf2py intent(hide) :: Nx, Ny
cf2py intent(out) :: q

      do j=1,Nx-1
        do i=1,Ny-1
          q(i, j) = ((uvh(i+Iv_i, j) + uvh(i+Iv_i, j+1)) / dx
     &             + (uvh(i, j)      + uvh(i+1, j)) / dy
     &             + f0)
     &             / (0.25*(h(i+1,j+1) + h(i+1,j) + h(i,j+1) + h(i,j)))
        enddo

c       last row of q, all columns (except last)
        q(Ny, j) = ((uvh(Iv_f, j) + uvh(Iv_f, j+1)) / dx
     &            + (uvh(Ny, j)   + uvh(1, j)) / dy
     &            + f0)
     &            / (0.25*(h(1,j+1) + h(1,j) + h(Ny,j+1) + h(i,j)))
      enddo

c     last column of q, all rows (except last)
      do i=1,Ny-1
        q(i, Nx) = ((uvh(i+Iv_i, Nx) + uvh(i+Iv_i, 1)) / dx
     &            + (uvh(i, Nx)      + uvh(i+1, Nx)) / dy
     &            + f0)
     &            / (0.25*(h(i+1,1) + h(i+1,Nx) + h(i,1) + h(i,j)))
      enddo

c     last column and row of q
      q(Ny, Nx) = ((uvh(Iv_f, Nx) + uvh(Iv_i+1, 1)) / dx
     &          +  (uvh(Ny, Nx)   + uvh(Ny, 1)) / dy
     &          + f0)
     &          / (0.25*(h(1,1) + h(1,Nx) + h(Ny,1) + h(Ny,Nx)))
      end