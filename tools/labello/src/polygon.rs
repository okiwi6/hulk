use eframe::{
    egui::{pos2, Color32, Mesh, Painter, Pos2, Rect, Stroke, TextureId},
    emath::RectTransform,
    epaint::{PathShape, Vertex, WHITE_UV},
};
use lyon_tessellation::{
    geometry_builder::simple_builder, math::Point, path::Path, FillOptions, FillTessellator,
    VertexBuffers,
};

#[derive(Debug, Default)]
pub struct FixedPolygon {
    border: Vec<Pos2>,

    vertices: Vec<Pos2>,
    indices: Vec<u16>,
    color: Color32,
}

#[derive(Debug, Default)]
pub struct FixedPolygonBuilder {
    points: Vec<Pos2>,
    color: Color32,
}

impl FixedPolygonBuilder {
    pub fn new(points: Vec<Pos2>) -> Self {
        Self {
            points,
            ..Default::default()
        }
    }

    pub fn color(mut self, color: Color32) -> Self {
        self.color = color;
        self
    }

    pub fn build(self) -> FixedPolygon {
        let (vertices, indices) = tessellate(&self.points);
        FixedPolygon {
            border: self.points,
            vertices,
            indices,
            color: self.color,
        }
    }
}

impl FixedPolygon {
    pub fn triangles(&self) -> Vec<[Pos2; 3]> {
        let identity = RectTransform::identity(Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)));
        let mesh = to_mesh(&self.vertices, &self.indices, identity, self.color);

        mesh.indices
            .chunks_exact(3)
            .map(|indices| {
                let a = mesh.vertices[indices[0] as usize].pos;
                let b = mesh.vertices[indices[1] as usize].pos;
                let c = mesh.vertices[indices[2] as usize].pos;
                [a, b, c]
            })
            .collect()
    }

    fn mesh(&self, transform: RectTransform) -> Mesh {
        to_mesh(&self.vertices, &self.indices, transform, self.color)
    }

    pub fn paint(&self, painter: &Painter, transform: RectTransform) {
        let mesh = self.mesh(transform);
        painter.add(mesh);
        painter.add(PathShape {
            points: self
                .border
                .iter()
                .map(|point| transform.transform_pos(*point))
                .collect(),
            closed: true,
            stroke: Stroke::new(3.0, self.color),
            fill: Color32::TRANSPARENT,
        });
    }
}

pub fn paint_polygon(
    painter: &Painter,
    points: impl Iterator<Item = Pos2>,
    transform: RectTransform,
    color: Color32,
) {
    let points = points.collect::<Vec<_>>();
    if points.len() >= 3 {
        let (vertices, indices) = tessellate(&points);
        let mesh = to_mesh(&vertices, &indices, transform, color.gamma_multiply(0.5));
        painter.add(mesh);
    }
    if points.len() >= 2 {
        // mal ne nlinies
        painter.add(PathShape {
            points: points
                .iter()
                .map(|point| transform.transform_pos(*point))
                .collect(),
            closed: points.len() > 2,
            stroke: Stroke::new(3.0, color),
            fill: Color32::TRANSPARENT,
        });
    }
}

fn tessellate(points: &[Pos2]) -> (Vec<Pos2>, Vec<u16>) {
    let path = {
        let mut path_builder = Path::builder();
        if let Some(first_point) = points.first() {
            path_builder.begin(convert(first_point));
        }
        for point in points.iter().skip(1) {
            path_builder.line_to(convert(point));
        }
        path_builder.end(true);
        path_builder.build()
    };
    let mut buffers: VertexBuffers<Point, u16> = VertexBuffers::new();
    {
        let mut vertex_builder = simple_builder(&mut buffers);

        // Create the tessellator.
        let mut tessellator = FillTessellator::new();

        // Compute the tessellation.
        tessellator
            .tessellate_path(
                &path,
                &FillOptions::default().with_tolerance(0.001),
                &mut vertex_builder,
            )
            .expect("failed to tessellate");
    }

    let vertices = buffers
        .vertices
        .into_iter()
        .map(|v| pos2(v.x, v.y))
        .collect();
    let indices = buffers.indices;

    (vertices, indices)
}
fn to_mesh(vertices: &[Pos2], indices: &[u16], transform: RectTransform, color: Color32) -> Mesh {
    let vertices = vertices
        .iter()
        .map(|v| Vertex {
            pos: transform.transform_pos(*v),
            uv: WHITE_UV,
            color,
        })
        .collect();
    let indices = indices.iter().map(|i| *i as u32).collect();

    Mesh {
        texture_id: TextureId::Managed(0),
        indices,
        vertices,
    }
}

fn convert(point: &Pos2) -> lyon_tessellation::math::Point {
    lyon_tessellation::math::Point::new(point.x, point.y)
}
