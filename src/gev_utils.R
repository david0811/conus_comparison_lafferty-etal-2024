library(extRemes)
library(parallel)
library(foreach)
library(doParallel)

print_cores <- function() {
  print(paste("Number of cores:", parallel::detectCores()))
}

# Improved function to fit GEV model for a single lat/lon grid point
fit_nonstat_gev_mle_single_point <- function(data_slice, ilat, ilon, itimes, times, periods_for_level, return_period_years) {
  # Initialize result containers
  param_results_main <- rep(-1234, 4)
  param_results_lower <- rep(-1234, 4)
  param_results_upper <- rep(-1234, 4)
  return_levels_main <- array(-1234, dim = c(length(periods_for_level), length(return_period_years)))
  return_levels_lower <- array(-1234, dim = c(length(periods_for_level), length(return_period_years)))
  return_levels_upper <- array(-1234, dim = c(length(periods_for_level), length(return_period_years)))
  
  # Check if all NaNs
  if (all(is.na(data_slice))) {
    return(list(
      param_results_main = param_results_main,
      param_results_lower = param_results_lower,
      param_results_upper = param_results_upper,
      return_levels_main = return_levels_main,
      return_levels_lower = return_levels_lower,
      return_levels_upper = return_levels_upper,
      ilat = ilat,
      ilon = ilon
    ))
  }
  
  # Variable to store the model fit
  gev_fit <- NULL
  
  # Fit GEV model with MLE - wrapped in tryCatch similar to serial version
  tryCatch({  
    # Main fit
    gev_fit <- fevd(data_slice, location.fun = ~itimes, method = "MLE", type = "GEV")
    param_results_main <- gev_fit$results$par
    
    # Confidence intervals
    params_ci <- ci(gev_fit, alpha = 0.05, type = c("parameter"))
    param_results_lower <- params_ci[, 1]
    param_results_upper <- params_ci[, 3]
    
  }, error = function(e) {
    # If MLE fails, try GMLE as fallback
    tryCatch({
      print(paste("MLE failed for lat", ilat, "lon", ilon, ":", e$message))
      # GMLE fit
      gev_fit <<- fevd(data_slice, location.fun = ~itimes, method = "GMLE", type = "GEV")
      param_results_main <<- gev_fit$results$par
      
      # Confidence intervals
      params_ci <- ci(gev_fit, alpha = 0.05, type = c("parameter"))
      param_results_lower <<- params_ci[, 1]
      param_results_upper <<- params_ci[, 3]
    }, error = function(e) {
      print(paste("GMLE also failed for lat", ilat, "lon", ilon, ":", e$message))
      # Both methods failed, keep default values
      param_results_main <<- rep(-1234, 4)
      param_results_lower <<- rep(-1234, 4)
      param_results_upper <<- rep(-1234, 4)
      return_levels_main <<- array(-1234, dim = c(length(periods_for_level), length(return_period_years)))
      return_levels_lower <<- array(-1234, dim = c(length(periods_for_level), length(return_period_years)))
      return_levels_upper <<- array(-1234, dim = c(length(periods_for_level), length(return_period_years)))
      # Since we're in an error handler, we need to use <<- to assign to parent environment variables
    })
  })
  
  # Check if we have a valid fit and if parameters are realistic
  if (!is.null(gev_fit) && (gev_fit$results$par[4] < -2 || gev_fit$results$par[4] > 2)) {
    print(paste("Unrealistic parameters for lat", ilat, "lon", ilon))
    # Try GMLE instead
    tryCatch({
      # GMLE fit
      gev_fit <- fevd(data_slice, location.fun = ~itimes, method = "GMLE", type = "GEV")
      param_results_main <- gev_fit$results$par
      
      # Confidence intervals
      params_ci <- ci(gev_fit, alpha = 0.05, type = c("parameter"))
      param_results_lower <- params_ci[, 1]
      param_results_upper <- params_ci[, 3]
    }, error = function(e) {    
      print(paste("GMLE failed for lat", ilat, "lon", ilon, ":", e$message))
      param_results_main <<- rep(-1234, 4)
      param_results_lower <<- rep(-1234, 4)
      param_results_upper <<- rep(-1234, 4)
      return_levels_main <<- array(-1234, dim = c(length(periods_for_level), length(return_period_years)))
      return_levels_lower <<- array(-1234, dim = c(length(periods_for_level), length(return_period_years)))
      return_levels_upper <<- array(-1234, dim = c(length(periods_for_level), length(return_period_years)))
    })
  }
  
  # Calculate return levels only if we have a valid fit
  if (!is.null(gev_fit) && !all(param_results_main == -1234)) {
    # Get return levels and confidence intervals
    return_period_time_indices <- match(return_period_years, times)
    
    for (iyear in 1:length(return_period_years)) {
      tryCatch({
        # Get year index
        return_period_index <- return_period_time_indices[iyear]
        v <- make.qcov(gev_fit, vals = list(mu1 = return_period_index))
        
        # Get main return level - Fix array assignment
        rls_main <- return.level(gev_fit, return.period = periods_for_level, qcov = v)
        return_levels_main[, iyear] <- rls_main
        
        # Get confidence intervals - Fix array assignment
        rls_ci <- ci(gev_fit, alpha = 0.05, type = c("return.level"),
                     return.period = periods_for_level, qcov = v)
        return_levels_lower[, iyear] <- rls_ci[, 1]
        return_levels_upper[, iyear] <- rls_ci[, 3]
      }, error = function(e) {
        print(paste("Return level calculation failed for year", return_period_years[iyear], 
                    "at lat", ilat, "lon", ilon, ":", e$message))
        return_levels_main[, iyear] <<- rep(-1234, length(periods_for_level))
        return_levels_lower[, iyear] <<- rep(-1234, length(periods_for_level))
        return_levels_upper[, iyear] <<- rep(-1234, length(periods_for_level))
      })
    }
  }
  
  return(list(
    param_results_main = param_results_main,
    param_results_lower = param_results_lower,
    param_results_upper = param_results_upper,
    return_levels_main = return_levels_main,
    return_levels_lower = return_levels_lower,
    return_levels_upper = return_levels_upper,
    ilat = ilat,
    ilon = ilon
  ))
}

# Updated main function with parallelization
fit_nonstat_gev_mle_parallel <- function(data,
                                         starting_year = 1950,
                                         periods_for_level = c(10, 25, 50, 100),
                                         return_period_years = c(1975, 2000, 2025, 2050, 2075, 2100),
                                         num_cores = 4) {
  # Load required packages
  if (!requireNamespace("parallel", quietly = TRUE)) stop("Package 'parallel' needed")
  if (!requireNamespace("foreach", quietly = TRUE)) stop("Package 'foreach' needed")
  if (!requireNamespace("doParallel", quietly = TRUE)) stop("Package 'doParallel' needed")
  
  # Get dimensions of data
  nlat <- dim(data)[2]
  nlon <- dim(data)[3]
  ntime <- dim(data)[1]
  times <- starting_year:(starting_year + ntime - 1)
  itimes <- 1:ntime
  
  # Prepare parallel environment
  cl <- parallel::makeCluster(min(num_cores, nlat * nlon))
  doParallel::registerDoParallel(cl)
  
  # Export needed variables and functions to each worker
  parallel::clusterExport(cl, c("times", "itimes", "periods_for_level", 
                                "return_period_years", "fit_nonstat_gev_mle_single_point"),
                          envir = environment())
  
  # Create list of all lat/lon combinations
  grid_points <- expand.grid(ilat = 1:nlat, ilon = 1:nlon)
  
  # Run the parallel computation
  results <- foreach::foreach(i = 1:nrow(grid_points), 
                              .packages = c("extRemes"),  # Load necessary package on each worker
                              .errorhandling = "pass") %dopar% {
                                
                                ilat <- grid_points$ilat[i]
                                ilon <- grid_points$ilon[i]
                                
                                # Extract data slice
                                data_slice <- data[, ilat, ilon]
                                
                                # Fit GEV model using the improved separate function that handles errors better
                                fit_nonstat_gev_mle_single_point(data_slice, ilat, ilon, itimes, times, 
                                                                periods_for_level, return_period_years)
                              }
  
  # Stop the cluster
  parallel::stopCluster(cl)
  
  # Initialize arrays to store results
  param_results_main <- array(NA, dim = c(nlat, nlon, 4))
  param_results_lower <- array(NA, dim = c(nlat, nlon, 4))
  param_results_upper <- array(NA, dim = c(nlat, nlon, 4))
  return_levels_main <- array(NA, dim = c(nlat, nlon, length(periods_for_level), length(return_period_years)))
  return_levels_lower <- array(NA, dim = c(nlat, nlon, length(periods_for_level), length(return_period_years)))
  return_levels_upper <- array(NA, dim = c(nlat, nlon, length(periods_for_level), length(return_period_years)))
  
  # Fill in results
  for (res in results) {
    # Skip any failed runs
    if (!is.list(res)) next
    
    ilat <- res$ilat
    ilon <- res$ilon
    
    param_results_main[ilat, ilon, ] <- res$param_results_main
    param_results_lower[ilat, ilon, ] <- res$param_results_lower
    param_results_upper[ilat, ilon, ] <- res$param_results_upper
    
    # Properly assign the return levels to the correct dimensions
    return_levels_main[ilat, ilon, , ] <- res$return_levels_main
    return_levels_lower[ilat, ilon, , ] <- res$return_levels_lower
    return_levels_upper[ilat, ilon, , ] <- res$return_levels_upper
  }
  
  return(list(
    param_results_main = param_results_main, 
    param_results_lower = param_results_lower, 
    param_results_upper = param_results_upper, 
    return_levels_main = return_levels_main, 
    return_levels_lower = return_levels_lower, 
    return_levels_upper = return_levels_upper
  ))
}

fit_nonstat_gev_mle <- function(data,
                                starting_year = 1950,
                                periods_for_level = c(10, 25, 50, 100),
                                return_period_years = c(1975, 2000, 2025, 2050, 2075, 2100)) {
  # Get dimensions of data
  nlat <- dim(data)[2]
  nlon <- dim(data)[3]
  ntime <- dim(data)[1]
  times <- starting_year:(starting_year + ntime - 1)
  itimes <- 1:ntime
  
  # Initialize vector to store results
  param_results_main <- array(NA, dim = c(nlat, nlon, 4))
  param_results_lower <- array(NA, dim = c(nlat, nlon, 4))
  param_results_upper <- array(NA, dim = c(nlat, nlon, 4))
  return_levels_main <- array(NA, dim = c(nlat, nlon, length(periods_for_level), length(return_period_years)))
  return_levels_lower <- array(NA, dim = c(nlat, nlon, length(periods_for_level), length(return_period_years)))
  return_levels_upper <- array(NA, dim = c(nlat, nlon, length(periods_for_level), length(return_period_years)))
  
  # Loop through each lat and lon
  for (ilat in 1:nlat) {
    for (ilon in 1:nlon) {
      data_slice <- data[, ilat, ilon]
      # Check if all NaNs
      if (all(is.na(data_slice))) {
        param_results_main[ilat, ilon, ] <- c(-1234, -1234, -1234, -1234)
        param_results_lower[ilat, ilon, ] <- c(-1234, -1234, -1234, -1234)
        param_results_upper[ilat, ilon, ] <- c(-1234, -1234, -1234, -1234)
        return_levels_main[ilat, ilon, , ] <- rep(-1234, length(periods_for_level) * length(return_period_years))
        return_levels_lower[ilat, ilon, , ] <- rep(-1234, length(periods_for_level) * length(return_period_years))
        return_levels_upper[ilat, ilon, , ] <- rep(-1234, length(periods_for_level) * length(return_period_years))
        next
      }
      
      # Fit GEV model
      tryCatch({  
        # Main fit
        gev_fit <- fevd(data_slice, location.fun = ~itimes, method = "MLE", type = "GEV")
        param_results_main[ilat, ilon, ] <- gev_fit$results$par
        # Confidence intervals
        params_ci <- ci(gev_fit, alpha = 0.05, type = c("parameter"))
        param_results_lower[ilat, ilon, ] <- params_ci[, 1]
        param_results_upper[ilat, ilon, ] <- params_ci[, 3]
        
      }, error = function(e) {
        tryCatch({
          print(e)
          # GMLE fit
          gev_fit <- fevd(data_slice, location.fun = ~itimes, method = "GMLE", type = "GEV")
          param_results_main[ilat, ilon, ] <- gev_fit$results$par
          # Confidence intervals
          params_ci <- ci(gev_fit, alpha = 0.05, type = c("parameter"))
          param_results_lower[ilat, ilon, ] <- params_ci[, 1]
          param_results_upper[ilat, ilon, ] <- params_ci[, 3]
        }, error = function(e) {
          print(e)
          param_results_main[ilat, ilon, ] <- c(-1234, -1234, -1234, -1234)
          param_results_lower[ilat, ilon, ] <- c(-1234, -1234, -1234, -1234)
          param_results_upper[ilat, ilon, ] <- c(-1234, -1234, -1234, -1234)
          return_levels_main[ilat, ilon, , ] <- rep(-1234, length(periods_for_level) * length(return_period_years))
          return_levels_lower[ilat, ilon, , ] <- rep(-1234, length(periods_for_level) * length(return_period_years))
          return_levels_upper[ilat, ilon, , ] <- rep(-1234, length(periods_for_level) * length(return_period_years))
        })
      })
      
      # If the returned parameters are unrealistic, try with GMLE
      if (gev_fit$results$par[4] < -2 || gev_fit$results$par[4] > 2) {
        print(paste("Unrealistic parameters for lat", ilat, "lon", ilon))
        tryCatch({
          # GMLE fit
          gev_fit <- fevd(data_slice, location.fun = ~itimes, method = "GMLE", type = "GEV")
          param_results_main[ilat, ilon, ] <- gev_fit$results$par
          # Confidence intervals
          params_ci <- ci(gev_fit, alpha = 0.05, type = c("parameter"))
          param_results_lower[ilat, ilon, ] <- params_ci[, 1]
          param_results_upper[ilat, ilon, ] <- params_ci[, 3]
        }, error = function(e) {    
          print(e)
          param_results_main[ilat, ilon, ] <- c(-1234, -1234, -1234, -1234)
          param_results_lower[ilat, ilon, ] <- c(-1234, -1234, -1234, -1234)
          param_results_upper[ilat, ilon, ] <- c(-1234, -1234, -1234, -1234)
          return_levels_main[ilat, ilon, , ] <- rep(-1234, length(periods_for_level) * length(return_period_years))
          return_levels_lower[ilat, ilon, , ] <- rep(-1234, length(periods_for_level) * length(return_period_years))
          return_levels_upper[ilat, ilon, , ] <- rep(-1234, length(periods_for_level) * length(return_period_years))
        })
      }
      
      # Get return levels and confidence intervals
      return_period_time_indices <- match(return_period_years, times)
      for (iyear in 1:length(return_period_years)) {
        tryCatch({
          # Get year index
          return_period_index <- return_period_time_indices[iyear]
          v <- make.qcov(gev_fit, vals = list(mu1 = return_period_index))
          # Get main return level
          rls_main <- return.level(gev_fit, return.period = periods_for_level, qcov = v)
          return_levels_main[ilat, ilon, , iyear] <- rls_main
          # Get confidence intervals
          rls_ci <- ci(gev_fit, alpha = 0.05, type = c("return.level"),
                       return.period = periods_for_level, qcov = v)
          return_levels_lower[ilat, ilon, , iyear] <- rls_ci[, 1]
          return_levels_upper[ilat, ilon, , iyear] <- rls_ci[, 3]
        }, error = function(e) {
          return_levels_main[ilat, ilon, , iyear] <- rep(-1234, length(periods_for_level))
          return_levels_lower[ilat, ilon, , iyear] <- rep(-1234, length(periods_for_level))
          return_levels_upper[ilat, ilon, , iyear] <- rep(-1234, length(periods_for_level))
        })
      }
    }
  }
  return(list(param_results_main = param_results_main, param_results_lower = param_results_lower, param_results_upper = param_results_upper, return_levels_main = return_levels_main, return_levels_lower = return_levels_lower, return_levels_upper = return_levels_upper))
}