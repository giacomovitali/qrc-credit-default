# Define the struct for CRC layer
Base.@kwdef struct ClassicalDetuningLayer
    atoms # atom positions
    Ω::Real # Rabi frequency
    t_start::Real # evolution starting time
    t_end::Real # evolution ending time
    step::Real  # readout time step
    state::Vector{<:Real} # classical state
    readout::String #readoud ("s" or "ss")
end


function generate_Vmat(locs::Vector{Vector{Float64}}, C6=862690*2π)
    nsites=length(locs)
    Vmat=zeros(nsites,nsites)
    VDel=zeros(nsites)
    for i in 1:nsites
        for j in 1:nsites
            if i != j
                Vmat[i,j]=C6/(norm(locs[i,:]-locs[j,:]))^6
                VDel[i]+=Vmat[i,j]
            end
        end
    end
    return Vmat, VDel
end

function deriv!(du, u, p::Tuple{Int, Vector{<:Real}, <:Real, Matrix{<:Real}, Vector{<:Real}},t)
    #calculating du/dt
    nsites, Delta, Omega, Vmat, VDel =p
    Bv=Vmat*u[3:3:3*nsites]
    for i in 1:nsites
        Bi = [Omega, 0, -Delta[i]+Bv[i]/2+VDel[i]/2]/2
        s = u[3*i-2:3*i]
        du[3*(i-1)+1] = Bi[3]*s[2] - Bi[2]*s[3]
        du[3*(i-1)+2] = Bi[1]*s[3] - Bi[3]*s[1]
        du[3*(i-1)+3] = Bi[2]*s[1] - Bi[1]*s[2] 
    end
    return du
end

function apply_classical_layer(layer::ClassicalDetuningLayer, x::Vector{<:Real})
    # define Rydberg hamiltonian, detunings parameterized in terms of pca values (x)  
    locs=layer.atoms
    readout=layer.readout
    nsites=length(locs)
    Vmat, VDel =generate_Vmat(locs) 
    # At the start of the simulation, all atoms of the system are in ground states
    u0 = layer.state

    t_start = layer.t_start
    t_end = layer.t_end
    t_step = layer.step
    
    # initialize output vector
    steps = floor(Int, (t_end - t_start) / t_step)
    out2=zeros(3*nsites, 3*nsites, steps)
    out3=zeros(nsites, nsites, steps)
    out31=zeros(div(nsites*(nsites-1),2), steps)
    # Numerically simulate the classical evolution with and store the results
    Omega=layer.Ω
    timespan=(t_start,t_end)
    prob = ODEProblem(deriv!,u0,timespan,(nsites,x,Omega,Vmat, VDel))
    sol = solve(prob, RK4(), saveat= t_step, adaptive=false, dt=1e-3, save_start=false)
    sol3=sol[3:3:3*nsites,:]
    if readout=="s"
        out=reduce(vcat, sol)
    elseif readout=="ss"
        for i in 1:steps
            out2[:,:,i]=sol[:,i] .* sol[:,i]'
        end
        out=reduce(vcat,(reduce(vcat,sol),reduce(vcat,out2)))
    elseif readout=="zz"
        for i in 1:steps
            ind=1
            for i1 in 1:nsites
                for i2 in (i1+1):nsites
                    out31[ind, i]= sol3[i1,i]* sol3[i2,i]
                    ind+=1
                end
            end
            #out3[:,:,i]=sol3[:,i] .* sol3[:,i]'
        end
        out=reduce(vcat,(reduce(vcat,sol3),reduce(vcat,out31)))
    elseif readout=="szz"
        ind=1
        for i in 1:steps
            for i1 in 1:nsites
                for i2 in (i1+1):nsites
                    out31[ind, i]= sol3[i1,i]* sol3[i2,i]
                    ind+=1
                end
            end
            #out3[:,:,i]=sol3[:,i] .* sol3[:,i]'
        end
        out=reduce(vcat,(reduce(vcat,sol),reduce(vcat,out3)))
    elseif readout=="z"
        out=reduce(vcat, sol3)
    end
    
    return out
end

# implement functions that apply a `DetuningLayer` to a matrix containing scaled detunings for each image
function apply_classical_layer(layer::ClassicalDetuningLayer, x::Matrix{<:Real})
    outs = [apply_classical_layer(layer, x[:, i][:]) for i in 1:size(x, 2)]
    return hcat(outs...)
end