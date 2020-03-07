use vulkano::device::{Device, DeviceExtensions};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;

use vulkano::sync;
use vulkano::sync::GpuFuture;

use std::sync::Arc;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::pipeline::ComputePipeline;

use rand::{thread_rng, Rng};

fn main() {
    let instance =
        Instance::new(None, &InstanceExtensions::none(), None).expect("failed to create instance");

    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");

    println!(
        "{:?}\n",
        PhysicalDevice::enumerate(&instance).collect::<Vec<_>>()
    );
    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_compute())
        .expect("couldn't find a graphical queue family");

    println!("{:?}\n", queue_family);

    let (device, mut queues) = {
        Device::new(
            physical,
            physical.supported_features(),
            &DeviceExtensions {
                khr_storage_buffer_storage_class: true,
                ..DeviceExtensions::none()
            },
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    let mut rng = thread_rng();

    let lower_bound = 0.0;
    let upper_bound = 100.0;
    let data_in_iter = (0..1024*100).map(|_| (rng.gen_range(lower_bound, upper_bound), rng.gen_range(lower_bound, upper_bound), rng.gen_range(1.0, 3.0)));
    let mut rng = thread_rng();
    let data_in_iter2 = (0..1024*100).map(|_| (rng.gen_range(lower_bound, upper_bound), rng.gen_range(lower_bound, upper_bound), rng.gen_range(1.0, 3.0)));
    for (x, y, r) in data_in_iter2 {
        println!("x: {} y: {} r: {}", x, y, r);
    }

    let data_in_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, data_in_iter)
            .expect("failed to create buffer");
            
    let data_out_iter = (0..1024*100).map(|_| -1);
    let data_out_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, data_out_iter)
            .expect("failed to create buffer");

    let compute_pipeline = Arc::new({
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                src:
                "
                    #version 450
                    
                    layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;
                    
                    layout(set = 0, binding = 0) buffer DataIn {
                        dvec3 data[];
                    } buf_in;
    
                    layout(set = 0, binding = 1) buffer DataOut {
                        uint data[];
                    } buf_out;
                    
                    void main() {
                        uint idx = gl_GlobalInvocationID.x;
                        double dist_squared = 
                            (buf_in.data[idx].x - 50) * (buf_in.data[idx].x - 50) +
                            (buf_in.data[idx].y - 50) * (buf_in.data[idx].y - 50);
                        buf_out.data[idx] = 
                            dist_squared <= (2 + buf_in.data[idx].z) * (2 + buf_in.data[idx].z) ? 1 : 0;
                    }
                "
            }
        }
        let shader = cs::Shader::load(device.clone()).expect("failed to create shader module");
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
            .expect("failed to create compute pipeline")
    });

    let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();
    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(data_in_buffer.clone())
            .unwrap()
            .add_buffer(data_out_buffer.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    let command_buffer =
        AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
            .unwrap()
            .dispatch([100, 1, 1], compute_pipeline.clone(), set.clone(), ())
            .unwrap()
            .build()
            .unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let content = data_out_buffer.read().unwrap();
    let mut n_colliding = 0;
    for (n, val) in content.iter().enumerate() {
        n_colliding += val;
        /* println!("{} {:?}", n, val); */
    }

    println!("{}", n_colliding);

    println!("Everything succeeded!");
}
